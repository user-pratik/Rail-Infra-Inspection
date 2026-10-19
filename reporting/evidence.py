"""Saves evidence images for a closed event (runs on the reporting worker thread)."""

from __future__ import annotations

from pathlib import Path

import cv2

SEV_BGR = {"critical": (40, 40, 198), "moderate": (0, 108, 239), "info": (37, 168, 249)}


def _to_bgr(img):
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _annotate(img, bbox, label, color):
    out = img.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ty = max(y1, th + 8)
    cv2.rectangle(out, (x1, ty - th - 8), (x1 + tw + 8, ty), color, -1)
    cv2.putText(out, label, (x1 + 4, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _crop(img, bbox, pad_frac):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    pad = max(10, int(pad_frac * max(x2 - x1, y2 - y1)))
    x1, y1, x2, y2 = max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]
    scale = min(6.0, max(1.0, 220.0 / max(crop.shape[:2])))
    if scale > 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop


def save_event_evidence(out_dir: Path, track, cfg, severity: str) -> list[str]:
    """Writes annotated.jpg, crop.jpg, strip_first.jpg, strip_best.jpg, strip_last.jpg."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    quality = int(cfg.data.get("evidence", {}).get("jpeg_quality", 90))
    pad = float(cfg.data.get("evidence", {}).get("crop_padding", 0.25))
    color = SEV_BGR.get(severity, (0, 108, 239))
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    label = f"{track.cls} {track.best_conf:.0%}"
    saved = []

    def write(name, img):
        if img is not None and cv2.imwrite(str(out_dir / name), img, params):
            saved.append(name)

    best = _to_bgr(track.best_img)
    if best is not None:
        write("annotated.jpg", _annotate(best, track.best_bbox, label, color))
        write("crop.jpg", _crop(best, track.best_bbox, pad))
    for name, img, bbox in (
        ("strip_first.jpg", _to_bgr(track.first_img), track.first_bbox),
        ("strip_best.jpg", best, track.best_bbox),
        ("strip_last.jpg", _to_bgr(track.last_img), track.last_bbox),
    ):
        if img is not None:
            write(name, _annotate(img, bbox, track.cls, color))
    return saved
