"""
Event aggregator.

Turns per-frame detections (which have no track IDs in the current codebase)
into one event per physical defect:

  * frame-to-frame association by IoU, falling back to centre distance
    (the camera moves, so a defect shifts between frames)
  * N-of-M confirmation: an event only counts once it was detected in at least
    N of the last M frames (removes one-frame flickers)
  * an event closes when it has not been seen for `max_gap_frames`
  * the highest-confidence frame is kept as evidence, plus first/last frames
  * optional same-frame box merging by IoU (e.g. merge_same_class_iou: 0.30)

Runs on the reporting worker thread, never on the inference thread.
"""

from __future__ import annotations

import datetime as dt
import math
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    cls: str            # canonical class key, e.g. "crack"
    conf: float
    bbox: tuple         # (x1, y1, x2, y2) in frame pixels


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _centre(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _diag(b) -> float:
    return math.hypot(b[2] - b[0], b[3] - b[1])


class Track:
    """One candidate/confirmed defect event while it is still open."""

    def __init__(self, tid: int, det: Detection, frame_idx: int, frame, window: int):
        self.tid = tid
        self.cls = det.cls
        self.created_wall = dt.datetime.now()
        self.first_frame = frame_idx
        self.last_frame = frame_idx
        self.last_seen = frame_idx
        self.bbox = det.bbox
        self.hits = 1
        self.conf_sum = det.conf
        self.best_conf = det.conf
        self.best_frame = frame_idx
        self.best_bbox = det.bbox
        self.recent = deque([True], maxlen=window)
        self.confirmed = False
        self.reported = False

        # Store frame dimensions when first seen for area-fraction checks
        if frame is not None and hasattr(frame, "shape") and len(frame.shape) >= 2:
            self.frame_h, self.frame_w = int(frame.shape[0]), int(frame.shape[1])
        else:
            self.frame_h, self.frame_w = None, None

        # evidence (references to worker-owned frame copies)
        self.first_img, self.first_bbox = frame, det.bbox
        self.best_img = frame
        self.last_img, self.last_bbox, self.last_img_frame = frame, det.bbox, frame_idx

    @property
    def best_area_frac(self) -> float | None:
        """Fraction of total frame area occupied by the best bounding box."""
        if self.frame_w and self.frame_h and self.best_bbox:
            bw = max(0, self.best_bbox[2] - self.best_bbox[0])
            bh = max(0, self.best_bbox[3] - self.best_bbox[1])
            return (bw * bh) / float(self.frame_w * self.frame_h)
        return None


class Aggregator:
    def __init__(self, agg_cfg: dict):
        self.confirm_hits = int(agg_cfg["confirm_hits"])
        self.window = int(agg_cfg["confirm_window"])
        self.max_gap = int(agg_cfg["max_gap_frames"])
        self.match_iou = float(agg_cfg["match_iou"])
        self.center_factor = float(agg_cfg["match_center_factor"])
        self.min_conf = float(agg_cfg["min_detection_confidence"])
        self.last_every = int(agg_cfg["last_image_every"])
        merge_iou = agg_cfg.get("merge_same_class_iou")
        self.merge_same_class_iou = float(merge_iou) if merge_iou is not None else None
        self.open: list[Track] = []
        self._next_id = 1

    # -- same-frame box merging ---------------------------------------------
    def _merge_same_class(self, dets: list[Detection]) -> list[Detection]:
        if not dets or self.merge_same_class_iou is None:
            return dets
        merged_all = []
        classes = set(d.cls for d in dets)
        for c in classes:
            c_dets = [d for d in dets if d.cls == c]
            changed = True
            while changed:
                changed = False
                new_dets = []
                used = set()
                for i in range(len(c_dets)):
                    if i in used:
                        continue
                    d1 = c_dets[i]
                    for j in range(i + 1, len(c_dets)):
                        if j in used:
                            continue
                        d2 = c_dets[j]
                        if _iou(d1.bbox, d2.bbox) >= self.merge_same_class_iou:
                            # Union box, max confidence
                            ubox = (
                                min(d1.bbox[0], d2.bbox[0]),
                                min(d1.bbox[1], d2.bbox[1]),
                                max(d1.bbox[2], d2.bbox[2]),
                                max(d1.bbox[3], d2.bbox[3]),
                            )
                            d1 = Detection(cls=c, conf=max(d1.conf, d2.conf), bbox=ubox)
                            used.add(j)
                            changed = True
                    new_dets.append(d1)
                c_dets = new_dets
            merged_all.extend(c_dets)
        return merged_all

    # -- matching -----------------------------------------------------------
    def _match_score(self, track_bbox, det_bbox) -> float:
        iou = _iou(track_bbox, det_bbox)
        if iou >= self.match_iou:
            return 1.0 + iou
        limit = self.center_factor * max(_diag(track_bbox), _diag(det_bbox), 1.0)
        (tx, ty), (dx, dy) = _centre(track_bbox), _centre(det_bbox)
        dist = math.hypot(tx - dx, ty - dy)
        return (1.0 - dist / limit) if dist <= limit else 0.0

    def _hit(self, t: Track, frame_idx: int, d: Detection, frame) -> None:
        t.last_seen = t.last_frame = frame_idx
        t.bbox = d.bbox
        t.hits += 1
        t.conf_sum += d.conf
        t.recent.append(True)
        if d.conf > t.best_conf:
            t.best_conf, t.best_frame, t.best_bbox, t.best_img = d.conf, frame_idx, d.bbox, frame
        if frame_idx - t.last_img_frame >= self.last_every:
            t.last_img, t.last_bbox, t.last_img_frame = frame, d.bbox, frame_idx
        if not t.confirmed and sum(t.recent) >= self.confirm_hits:
            t.confirmed = True

    # -- per-frame update ---------------------------------------------------
    def update(self, frame_idx: int, dets: list[Detection], frame) -> list[Track]:
        """Advance one frame. Returns confirmed tracks that just closed."""
        dets = [d for d in dets if d.conf >= self.min_conf]
        if self.merge_same_class_iou is not None:
            dets = self._merge_same_class(dets)

        pairs = []
        for ti, t in enumerate(self.open):
            for di, d in enumerate(dets):
                if d.cls != t.cls:
                    continue
                s = self._match_score(t.bbox, d.bbox)
                if s > 0:
                    pairs.append((s, ti, di))
        pairs.sort(reverse=True)

        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            self._hit(self.open[ti], frame_idx, dets[di], frame)

        for ti, t in enumerate(self.open):
            if ti not in used_t:
                t.recent.append(False)

        for di, d in enumerate(dets):
            if di not in used_d:
                t = Track(self._next_id, d, frame_idx, frame, self.window)
                self._next_id += 1
                if self.confirm_hits <= 1:
                    t.confirmed = True
                self.open.append(t)

        closed, still_open = [], []
        for t in self.open:
            if frame_idx - t.last_seen > self.max_gap:
                if t.confirmed and not t.reported:
                    closed.append(t)
            else:
                still_open.append(t)
        self.open = still_open
        return closed

    def flush(self) -> list[Track]:
        """Close everything (end of video / user pressed Stop)."""
        closed = [t for t in self.open if t.confirmed and not t.reported]
        self.open = []
        return closed

    def flush_confirmed(self) -> list[Track]:
        """Snapshot/flush all confirmed tracks without dropping them from open."""
        to_report = []
        for t in self.open:
            if t.confirmed and not t.reported:
                t.reported = True
                to_report.append(t)
        return to_report
