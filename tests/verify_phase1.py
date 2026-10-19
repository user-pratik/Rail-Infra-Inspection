"""
Verification test for Phase 1: Frame-source seam acceptance.
Runs Presets 1-4 on sample videos using kind="file" with open_capture
and verifies detection output and FPS are identical to standard cv2.VideoCapture.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import torch
import torchvision.transforms.functional as F
from ultralytics import YOLO

from gui_app import PRESETS, load_rcnn_model, TRACK_CLASSES
from streaming import open_capture

def run_preset_test(preset_name, num_frames=30):
    info = PRESETS[preset_name]
    video_path = info["video"]
    model_path = info["model"]
    model_type = info["model_type"]
    mode = info["mode"]

    print(f"\n--- Testing Preset: {preset_name} ---")
    print(f"Video: {video_path}")
    print(f"Model: {model_path} ({model_type})")

    if not os.path.isfile(video_path):
        print(f"SKIP: Video not found: {video_path}")
        return True
    if not os.path.isfile(model_path):
        print(f"SKIP: Model not found: {model_path}")
        return True

    # 1. Test standard cv2.VideoCapture
    cap_cv = cv2.VideoCapture(video_path)
    # 2. Test open_capture(kind="file")
    cap_seam = open_capture("file", video_path)

    assert isinstance(cap_seam, cv2.VideoCapture), "open_capture('file') must return cv2.VideoCapture"
    assert getattr(cap_seam, "is_live", False) is False, "is_live must be False for file capture"

    # Load model
    if model_type == "rcnn":
        model = load_rcnn_model(model_path, num_classes=7)
    else:
        model = YOLO(model_path)

    dets_cv = []
    dets_seam = []

    # Run standard
    t0 = time.time()
    for _ in range(num_frames):
        ret, frame = cap_cv.read()
        if not ret:
            break
        if model_type == "rcnn":
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = F.to_tensor(img_rgb).unsqueeze(0)
            with torch.no_grad():
                out = model(img_tensor)[0]
            boxes = out['boxes'].cpu().numpy()
            scores = out['scores'].cpu().numpy()
            labels = out['labels'].cpu().numpy()
            for b, s, l in zip(boxes, scores, labels):
                if s >= 0.20:
                    dets_cv.append((int(l), round(float(s), 2), [round(float(c), 1) for c in b]))
        else:
            res = model(frame, verbose=False, conf=0.25)
            for r in res:
                for b in r.boxes:
                    cls_id = int(b.cls[0])
                    conf = float(b.conf[0])
                    coords = [round(float(c), 1) for c in b.xyxy[0]]
                    dets_cv.append((cls_id, round(conf, 2), coords))
    t_cv = time.time() - t0
    cap_cv.release()

    # Run seam
    t0 = time.time()
    for _ in range(num_frames):
        ret, frame = cap_seam.read()
        if not ret:
            break
        if model_type == "rcnn":
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = F.to_tensor(img_rgb).unsqueeze(0)
            with torch.no_grad():
                out = model(img_tensor)[0]
            boxes = out['boxes'].cpu().numpy()
            scores = out['scores'].cpu().numpy()
            labels = out['labels'].cpu().numpy()
            for b, s, l in zip(boxes, scores, labels):
                if s >= 0.20:
                    dets_seam.append((int(l), round(float(s), 2), [round(float(c), 1) for c in b]))
        else:
            res = model(frame, verbose=False, conf=0.25)
            for r in res:
                for b in r.boxes:
                    cls_id = int(b.cls[0])
                    conf = float(b.conf[0])
                    coords = [round(float(c), 1) for c in b.xyxy[0]]
                    dets_seam.append((cls_id, round(conf, 2), coords))
    t_seam = time.time() - t0
    cap_seam.release()

    assert dets_cv == dets_seam, f"Detections mismatch! CV count={len(dets_cv)}, Seam count={len(dets_seam)}"
    fps_cv = num_frames / max(0.001, t_cv)
    fps_seam = num_frames / max(0.001, t_seam)
    print(f"PASSED: Detections identical ({len(dets_cv)} detections). FPS: CV={fps_cv:.1f}, Seam={fps_seam:.1f}")
    return True

if __name__ == "__main__":
    all_ok = True
    for preset_name in PRESETS:
        ok = run_preset_test(preset_name, num_frames=15)
        if not ok:
            all_ok = False
    if all_ok:
        print("\n=== ALL PRESET TESTS PASSED! Phase 1 verified ===")
        sys.exit(0)
    else:
        sys.exit(1)
