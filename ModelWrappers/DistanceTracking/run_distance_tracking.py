"""
Obstacle Detection and Real-Time Distance Estimation Engine
Calculates focal length and real-time distance in cm for railway obstacles with proximity collision alerts.
"""

import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

# Real-world estimated object widths in centimeters
KNOWN_WIDTH = {
    'Animal': 100,
    'Car': 180,
    'Person': 45,
    'Tree': 60,
    'Rock': 50,
    'Trash': 40
}

# Distance warning threshold in cm (e.g. 1500 cm = 15 meters)
WARNING_DISTANCE_CM = 1500.0


def calibrate_focal_length(known_distance_cm: float, real_width_cm: float, width_in_pixels: float) -> float:
    """Calibrate camera focal length using a reference detection."""
    return (width_in_pixels * known_distance_cm) / real_width_cm


def calculate_distance(focal_length: float, real_width_cm: float, width_in_pixels: float) -> float:
    """Calculate distance from camera to detected object in cm."""
    if width_in_pixels <= 0:
        return 9999.0
    return (real_width_cm * focal_length) / width_in_pixels


def run_video_tracking(
    model_path: str,
    video_path: str,
    output_path: str = None,
    calibration_image_path: str = None,
    max_frames: int = None,
    gui_mode: bool = True,
    conf_threshold: float = 0.25
):
    print(f"Loading YOLOv11 obstacle detection model: {model_path}")
    model = YOLO(model_path)

    # Adaptive focal length determination
    scaled_focal_length = 800.0 * (max(frame_width, 640) / 640.0)
    if calibration_image_path and os.path.exists(calibration_image_path):
        cal_img = cv2.imread(calibration_image_path)
        if cal_img is not None:
            results = model(cal_img, verbose=False, conf=conf_threshold)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    name = model.names[cls_id]
                    if name in KNOWN_WIDTH:
                        x1, y1, x2, y2 = box.xyxy[0]
                        w_px = float(x2 - x1)
                        if w_px > 10:
                            scaled_focal_length = calibrate_focal_length(700.0, KNOWN_WIDTH[name], w_px)
                            print(f"Calibrated focal length on class '{name}': {scaled_focal_length:.2f}px")
                            break

    print(f"Using scaled focal length: {scaled_focal_length:.2f}px (frame resolution: {frame_width}x{frame_height})")

    frame_idx = 0
    detections_summary = []
    paused = False

    window_name = "Railway Infrastructure Inspection - Obstacle & Distance Tracker"
    if gui_mode:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(1280, frame_width), min(720, frame_height))

    print(f"Processing {'all' if not frames_to_process else frames_to_process} frames from {video_path}...")
    if gui_mode:
        print("GUI Controls: [SPACE] Pause/Resume | [S] Step | [R] Restart | [Q / ESC] Quit")

    try:
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frames_to_process and frame_idx > frames_to_process:
                    break

                results = model(frame, verbose=False, conf=conf_threshold)
                frame_alert = False
                lateral_warning = False
                closest_dist = float("inf")
                obstacle_count = 0

                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = model.names[cls_id]

                        width_px = max(1, x2 - x1)
                        real_w = KNOWN_WIDTH.get(class_name, 60)
                        dist_cm = calculate_distance(scaled_focal_length, real_w, width_px)
                        dist_m = dist_cm / 100.0
                        obstacle_count += 1

                        cx = (x1 + x2) / 2.0
                        norm_cx = cx / float(frame_width)
                        norm_w = width_px / float(frame_width)
                        is_central_corridor = (0.32 <= norm_cx <= 0.68)
                        is_tunnel_or_wall = (norm_w > 0.40 and (x1 <= 50 or x2 >= frame_width - 50)) or not is_central_corridor

                        if is_tunnel_or_wall or not is_central_corridor:
                            # Yellow / Amber for Lateral / Tunnel Clearance (Safe Margin)
                            lateral_warning = True
                            box_color = (0, 215, 255)  # Amber / Yellow
                            label = f"{class_name} {conf*100:.0f}% | {dist_m:.1f}m (Clearance)"
                        else:
                            if dist_cm < closest_dist:
                                closest_dist = dist_cm
                            is_close = dist_cm < WARNING_DISTANCE_CM
                            if is_close:
                                box_color = (0, 0, 255)       # Red
                                frame_alert = True
                                label = f"{class_name} {conf*100:.0f}% | {dist_m:.1f}m [IN-TRACK]"
                            elif dist_cm < 3500.0:
                                box_color = (0, 165, 255)     # Amber
                                label = f"{class_name} {conf*100:.0f}% | {dist_m:.1f}m [APPROACHING]"
                            else:
                                box_color = (0, 255, 0)       # Green
                                label = f"{class_name} {conf*100:.0f}% | {dist_m:.1f}m [SAFE]"

                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                        badge_y1 = max(0, y1 - th - 10)
                        cv2.rectangle(frame, (x1, badge_y1), (x1 + tw + 10, y1), box_color, -1)
                        cv2.putText(
                            frame, label, (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2
                        )

                        detections_summary.append({
                            "frame": frame_idx,
                            "class": class_name,
                            "confidence": round(conf, 3),
                            "distance_m": round(dist_m, 1),
                            "in_track": is_central_corridor and not is_tunnel_or_wall,
                            "warning": is_close if (is_central_corridor and not is_tunnel_or_wall) else False
                        })

                # Top HUD Bar
                hud_overlay = frame.copy()
                cv2.rectangle(hud_overlay, (0, 0), (frame_width, 60), (20, 20, 20), -1)
                cv2.addWeighted(hud_overlay, 0.7, frame, 0.3, 0, frame)

                cv2.putText(frame, f"FRAME: {frame_idx}/{total_frames}", (20, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.putText(frame, f"DETECTIONS: {obstacle_count}", (200, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if obstacle_count > 0 else (200, 200, 200), 2)
                if closest_dist < float("inf"):
                    dist_text = f"TRACK OBSTACLE DIST: {closest_dist/100:.1f} m"
                    cv2.putText(frame, dist_text, (400, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if frame_alert else (0, 255, 0), 2)
                elif lateral_warning:
                    cv2.putText(frame, "TUNNEL / LATERAL CLEARANCE MAINTAINED", (400, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)

                if frame_alert:
                    alert_banner = frame.copy()
                    cv2.rectangle(alert_banner, (0, 60), (frame_width, 110), (0, 0, 180), -1)
                    cv2.addWeighted(alert_banner, 0.8, frame, 0.2, 0, frame)
                    cv2.putText(
                        frame, "CRITICAL ALERT: OBSTACLE DIRECTLY ON TRACK (< 15m)!",
                        (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2
                    )

                cv2.putText(frame, "[SPACE] Pause | [S] Step | [R] Restart | [Q] Quit",
                            (20, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

                if out is not None:
                    out.write(frame)

            if gui_mode:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(30 if not paused else 100) & 0xFF
                if key in (ord('q'), ord('Q'), 27):  # 'q' or ESC
                    print("Exiting GUI...")
                    break
                elif key == 32:  # Spacebar
                    paused = not paused
                elif key in (ord('s'), ord('S')) and paused:
                    # Single step
                    ret, frame = cap.read()
                    if ret:
                        frame_idx += 1
                        results = model(frame, verbose=False, conf=conf_threshold)
                        # Redraw frame for stepping
                        for r in results:
                            for box in r.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cls_id = int(box.cls[0])
                                conf = float(box.conf[0])
                                class_name = model.names[cls_id]
                                width_px = x2 - x1
                                real_w = KNOWN_WIDTH.get(class_name, 60)
                                dist_cm = calculate_distance(focal_length, real_w, width_px)
                                is_close = dist_cm < WARNING_DISTANCE_CM
                                box_color = (0, 0, 255) if is_close else (0, 255, 0)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                                label = f"{class_name} {conf*100:.0f}% | {dist_cm/100:.1f}m"
                                cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)
                        cv2.imshow(window_name, frame)
                elif key in (ord('r'), ord('R')):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx = 0
                    paused = False

    finally:
        cap.release()
        if out is not None:
            out.release()
        if gui_mode:
            cv2.destroyAllWindows()

    if output_path:
        print(f"Video processing finished. Annotated video saved to: {output_path}")
    return detections_summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Railway Obstacle Detection and Distance Tracking Engine")
    parser.add_argument("--model", type=str, default=None, help="Path to YOLO model weights")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file")
    parser.add_argument("--output", type=str, default=None, help="Path to output annotated video file")
    parser.add_argument("--no-gui", action="store_true", help="Run in headless mode without GUI window")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_file = args.model or os.path.join(BASE_DIR, "best_99).pt")
    video_file = args.video or os.path.join(BASE_DIR, "sample_railway_video.mp4")
    out_file = args.output

    run_video_tracking(
        model_path=model_file,
        video_path=video_file,
        output_path=out_file,
        max_frames=args.max_frames,
        gui_mode=not args.no_gui,
        conf_threshold=args.conf
    )
