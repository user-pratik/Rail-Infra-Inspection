"""
Railway Infrastructure Inspection System - Interactive GUI Dashboard
Provides real-time visual inspection:
 1. Obstacle Detection & Real-Time Distance Estimation with Track Corridor & Tunnel Clearance (YOLOv11)
 2. Railway Track Defect Detection (Faster R-CNN ResNet-50 FPN)
 3. Bridge Structural Defect Detection (YOLOv11)
"""

import os
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import torch
import torchvision.transforms.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ultralytics import YOLO
from reporting.hub import ReportingHub
from reporting.gui_panel import ReportingPanel
from streaming import open_capture

# Real-world estimated obstacle widths in centimeters
KNOWN_WIDTH = {
    'Animal': 100,
    'Car': 180,
    'Person': 50,
    'Tree': 80,
    'Rock': 60,
    'Trash': 50
}
WARNING_DISTANCE_CM = 1500.0  # 15 meters braking distance threshold
BASE_FOCAL_LENGTH_640 = 800.0  # Calibrated for 640px wide reference

# Track Defect class mapping from trained COCO dataset
TRACK_CLASSES = {
    1: 'Crack',
    2: 'Flaking',
    3: 'Groove',
    4: 'Joint',
    5: 'Spalling',
    6: 'Squat'
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PRESETS = {
    "Obstacle & Tunnel Clearance (track_vid_no_obs.mp4)": {
        "video": os.path.join(BASE_DIR, "track_vid_no_obs.mp4"),
        "model": os.path.join(BASE_DIR, "ModelWrappers", "DistanceTracking", "best_99).pt"),
        "model_type": "yolo",
        "mode": "distance_tracking",
        "desc": "Analyzes railway track corridor, tunnel entrances & lateral clearances on track_vid_no_obs.mp4"
    },
    "Obstacle Detection & Distance Estimation (sample_railway_video.mp4)": {
        "video": os.path.join(BASE_DIR, "ModelWrappers", "DistanceTracking", "sample_railway_video.mp4"),
        "model": os.path.join(BASE_DIR, "ModelWrappers", "DistanceTracking", "best_99).pt"),
        "model_type": "yolo",
        "mode": "distance_tracking",
        "desc": "Detects track obstacles (Person, Animal, Tree, Car) and estimates real-time distance"
    },
    "Railway Track Defect Inspection (Faster R-CNN)": {
        "video": os.path.join(BASE_DIR, "ModelWrappers", "TrackDefects", "sample_track_video.mp4"),
        "model": os.path.join(BASE_DIR, "ModelWrappers", "TrackDefects", "RCNN", "fasterrcnn_resnet50_fpn.pth"),
        "model_type": "rcnn",
        "mode": "track_defects",
        "report_module": "track",
        "desc": "Detects structural track defects: Cracks, Squats, Spalling, Joints, Flakings, Grooves"
    },
    "Bridge Structural Defect Detection (YOLOv11)": {
        "video": os.path.join(BASE_DIR, "ModelWrappers", "BridgeDefects", "sample_bridge_video.mp4"),
        "model": os.path.join(BASE_DIR, "ModelWrappers", "BridgeDefects", "YOLOV11", "weights", "best.pt"),
        "model_type": "yolo",
        "mode": "bridge_defects",
        "report_module": "bridge",
        "desc": "Inspects bridge infrastructure for Corrosion, Cracks, Freelime, Leakage, Spalling"
    }
}


def calculate_distance(focal_length: float, real_width_cm: float, width_in_pixels: float) -> float:
    if width_in_pixels <= 0:
        return 9999.0
    return (real_width_cm * focal_length) / width_in_pixels


def load_rcnn_model(model_path: str, num_classes: int = 7):
    """Load Faster R-CNN ResNet-50 FPN model with custom classifier head."""
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    ckpt = torch.load(model_path, map_location='cpu')
    model.load_state_dict(ckpt)
    model.eval()
    return model


class RailwayInspectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Railway Infrastructure & Defect Inspection System")
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        win_w = min(1280, max(1000, screen_w - 60))
        win_h = min(880, max(680, screen_h - 80))
        pos_x = max(0, (screen_w - win_w) // 2)
        pos_y = max(0, (screen_h - win_h) // 2)
        self.root.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
        self.root.minsize(1000, 680)
        self.root.configure(bg="#1a1b23")

        self.current_model = None
        self.current_model_path = ""
        self.current_model_type = ""
        self.cap = None
        self.is_running = False
        self.is_paused = False
        self.fps_timer = time.time()
        self.fps_count = 0
        self.current_fps = 0.0

        self.photo = None
        self.reporter = ReportingHub()
        self._report_token = 0
        self.playback_speed = 1.0
        self.capture_kind = "file"

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._on_preset_change(None)

    def _build_ui(self):
        # Header Bar
        header_frame = tk.Frame(self.root, bg="#12131a", height=65)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)

        title_box = tk.Frame(header_frame, bg="#12131a")
        title_box.pack(side=tk.LEFT, padx=20, pady=8)

        tk.Label(
            title_box,
            text="🚄 RAILWAY INFRASTRUCTURE INSPECTION SUITE",
            font=("Segoe UI", 15, "bold"),
            fg="#00e5ff",
            bg="#12131a"
        ).pack(anchor=tk.W)

        self.engine_badge = tk.Label(
            title_box,
            text="MODEL: YOLOv11 (Obstacle & Clearance Engine)",
            font=("Segoe UI", 8, "bold"),
            fg="#a0aec0",
            bg="#12131a"
        )
        self.engine_badge.pack(anchor=tk.W)

        self.status_badge = tk.Label(
            header_frame,
            text="STATUS: READY",
            font=("Segoe UI", 10, "bold"),
            bg="#2d3748",
            fg="#e2e8f0",
            padx=14,
            pady=6,
            relief=tk.FLAT
        )
        self.status_badge.pack(side=tk.RIGHT, padx=20, pady=14)

        # Main Layout: Left (Video Canvas), Right (Controls & Real-Time Stats)
        main_frame = tk.Frame(self.root, bg="#1a1b23")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=12)

        # Left Column (Video Screen + Live Alert Banner)
        left_col = tk.Frame(main_frame, bg="#1a1b23")
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))

        self.video_canvas = tk.Canvas(
            left_col,
            bg="#0b0c10",
            highlightthickness=2,
            highlightbackground="#2d3748"
        )
        self.video_canvas.pack(fill=tk.BOTH, expand=True)

        # Dynamic Status / Alert Banner
        self.alert_label = tk.Label(
            left_col,
            text="✅ System Initialized - Ready to start inspection",
            font=("Segoe UI", 11, "bold"),
            bg="#1c2d27",
            fg="#48bb78",
            pady=8
        )
        self.alert_label.pack(fill=tk.X, pady=(8, 0))

        # Right Column (Module Selector, Controls & Telemetry)
        right_col = tk.Frame(main_frame, bg="#1a1b23", width=400)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(0, 0))
        right_col.pack_propagate(False)

        # 1. Preset Selector Card
        preset_card = tk.LabelFrame(
            right_col,
            text=" ⚙️ Inspection Module ",
            font=("Segoe UI", 10, "bold"),
            bg="#232634",
            fg="#e2e8f0",
            padx=12,
            pady=10
        )
        preset_card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(preset_card, text="Select Inspection Pipeline:", font=("Segoe UI", 9), bg="#232634", fg="#a0aec0").pack(anchor=tk.W)
        self.preset_var = tk.StringVar(value=list(PRESETS.keys())[0])
        preset_dropdown = ttk.Combobox(
            preset_card,
            textvariable=self.preset_var,
            values=list(PRESETS.keys()),
            state="readonly",
            font=("Segoe UI", 9)
        )
        preset_dropdown.pack(fill=tk.X, pady=(4, 6))
        preset_dropdown.bind("<<ComboboxSelected>>", self._on_preset_change)

        self.lbl_module_desc = tk.Label(
            preset_card,
            text="",
            font=("Segoe UI", 8, "italic"),
            bg="#232634",
            fg="#718096",
            wraplength=360,
            justify=tk.LEFT
        )
        self.lbl_module_desc.pack(anchor=tk.W, pady=(0, 6))

        btn_custom = tk.Button(
            preset_card,
            text="📁 Select Custom Video File...",
            font=("Segoe UI", 9),
            bg="#3b4252",
            fg="#eceff4",
            activebackground="#4c566a",
            relief=tk.FLAT,
            command=self._choose_custom_video
        )
        btn_custom.pack(fill=tk.X, pady=(0, 4))

        # 2. Controls Card
        ctrl_card = tk.LabelFrame(
            right_col,
            text=" 🎮 Playback & Inference Controls ",
            font=("Segoe UI", 10, "bold"),
            bg="#232634",
            fg="#e2e8f0",
            padx=12,
            pady=10
        )
        ctrl_card.pack(fill=tk.X, pady=(0, 10))

        btn_row = tk.Frame(ctrl_card, bg="#232634")
        btn_row.pack(fill=tk.X, pady=(0, 8))

        self.btn_start = tk.Button(
            btn_row,
            text="▶ START",
            font=("Segoe UI", 10, "bold"),
            bg="#2b6cb0",
            fg="white",
            activebackground="#2c5282",
            relief=tk.FLAT,
            command=self.start_playback
        )
        self.btn_start.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_pause = tk.Button(
            btn_row,
            text="⏸ PAUSE",
            font=("Segoe UI", 10, "bold"),
            bg="#4a5568",
            fg="white",
            activebackground="#2d3748",
            state=tk.DISABLED,
            relief=tk.FLAT,
            command=self.toggle_pause
        )
        self.btn_pause.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        self.btn_stop = tk.Button(
            btn_row,
            text="⏹ STOP",
            font=("Segoe UI", 10, "bold"),
            bg="#9b2c2c",
            fg="white",
            activebackground="#742a2a",
            state=tk.DISABLED,
            relief=tk.FLAT,
            command=self.stop_playback
        )
        self.btn_stop.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        # Report & Digest Buttons in Playback Controls (Mid-Flow & Post-Run)
        report_btn_row = tk.Frame(ctrl_card, bg="#232634")
        report_btn_row.pack(fill=tk.X, pady=(0, 6))

        self.btn_daily_digest = tk.Button(
            report_btn_row,
            text="📑 Daily Digest",
            font=("Segoe UI", 9, "bold"),
            bg="#2c5282",
            fg="white",
            activebackground="#2b6cb0",
            relief=tk.FLAT,
            padx=6,
            pady=4,
            cursor="hand2",
            command=lambda: self.reporter.generate_digest(auto_open=True)
        )
        self.btn_daily_digest.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))

        self.btn_compile_report = tk.Button(
            report_btn_row,
            text="📄 Report",
            font=("Segoe UI", 9, "bold"),
            bg="#319795",
            fg="white",
            activebackground="#285e61",
            relief=tk.FLAT,
            padx=6,
            pady=4,
            cursor="hand2",
            command=lambda: self.reporter.compile_report(auto_open=True)
        )
        self.btn_compile_report.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))

        # Confidence Slider
        slider_row = tk.Frame(ctrl_card, bg="#232634")
        slider_row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(slider_row, text="Detection Confidence Threshold:", font=("Segoe UI", 9), bg="#232634", fg="#a0aec0").pack(side=tk.LEFT)
        self.conf_val_lbl = tk.Label(slider_row, text="0.25", font=("Segoe UI", 9, "bold"), bg="#232634", fg="#00e5ff")
        self.conf_val_lbl.pack(side=tk.RIGHT)

        self.conf_slider = ttk.Scale(
            ctrl_card,
            from_=0.10,
            to=0.95,
            value=0.25,
            command=lambda v: self.conf_val_lbl.config(text=f"{float(v):.2f}")
        )
        self.conf_slider.pack(fill=tk.X, pady=(2, 4))

        # Playback Speed Slider & Controls (0.5x to 4.0x)
        speed_row = tk.Frame(ctrl_card, bg="#232634")
        speed_row.pack(fill=tk.X, pady=(6, 0))
        tk.Label(speed_row, text="Playback Speed:", font=("Segoe UI", 9), bg="#232634", fg="#a0aec0").pack(side=tk.LEFT)
        self.speed_val_lbl = tk.Label(speed_row, text="1.0x", font=("Segoe UI", 9, "bold"), bg="#232634", fg="#00e5ff")
        self.speed_val_lbl.pack(side=tk.RIGHT)

        self.speed_slider = ttk.Scale(
            ctrl_card,
            from_=0.5,
            to=4.0,
            value=1.0,
            command=self._on_speed_change
        )
        self.speed_slider.pack(fill=tk.X, pady=(2, 2))

        speed_btn_row = tk.Frame(ctrl_card, bg="#232634")
        speed_btn_row.pack(fill=tk.X, pady=(0, 4))
        for spd in [0.5, 1.0, 2.0, 3.0, 4.0]:
            btn_spd = tk.Button(
                speed_btn_row,
                text=f"{spd}x",
                font=("Segoe UI", 8),
                bg="#3b4252",
                fg="#eceff4",
                activebackground="#4c566a",
                relief=tk.FLAT,
                command=lambda s=spd: self._set_speed(s)
            )
            btn_spd.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)

        # 3. Telemetry & Metrics Card
        metrics_card = tk.LabelFrame(
            right_col,
            text=" 📊 Real-Time Telemetry & Insights ",
            font=("Segoe UI", 10, "bold"),
            bg="#232634",
            fg="#e2e8f0",
            padx=12,
            pady=10
        )
        metrics_card.pack(fill=tk.X, pady=(0, 6))

        self.lbl_fps = self._create_metric_row(metrics_card, "Processing FPS:", "0.0 FPS")
        self.lbl_frame = self._create_metric_row(metrics_card, "Frame Counter:", "0 / 0")
        self.lbl_objects = self._create_metric_row(metrics_card, "Detections in Frame:", "0")
        
        self.metric_row_frame = tk.Frame(metrics_card, bg="#232634")
        self.metric_row_frame.pack(fill=tk.X, pady=2)
        self.lbl_metric_name = tk.Label(self.metric_row_frame, text="Key Finding:", font=("Segoe UI", 9), bg="#232634", fg="#a0aec0")
        self.lbl_metric_name.pack(side=tk.LEFT)
        self.lbl_metric_val = tk.Label(self.metric_row_frame, text="Normal", font=("Segoe UI", 9, "bold"), bg="#232634", fg="#ffffff")
        self.lbl_metric_val.pack(side=tk.RIGHT)

        # Detection Event Log
        tk.Label(metrics_card, text="Live Inspection Log:", font=("Segoe UI", 9, "bold"), bg="#232634", fg="#a0aec0").pack(anchor=tk.W, pady=(4, 2))
        
        self.log_listbox = tk.Listbox(
            metrics_card,
            height=4,
            bg="#151720",
            fg="#e2e8f0",
            font=("Consolas", 8),
            highlightthickness=0,
            selectbackground="#2b6cb0"
        )
        self.log_listbox.pack(fill=tk.X, pady=(0, 2))

        # 4. Automated Reporting Panel
        self.report_panel = ReportingPanel(
            right_col,
            self.reporter,
            self.root,
            log_fn=self._report_log,
            bg="#232634",
            fg="#e2e8f0",
            accent="#2b6cb0",
            font=("Segoe UI", 9)
        )
        self.report_panel.pack(fill=tk.X, pady=(0, 4))

        # Add manual Clear button to reporting panel button row
        try:
            panel_btn_row = self.report_panel.frame.winfo_children()[-1]
            tk.Button(
                panel_btn_row,
                text="Clear",
                command=self._clear_report_stats,
                bg="#4a5568",
                fg="white",
                relief="flat",
                font=("Segoe UI", 9),
                padx=8,
                pady=3,
                cursor="hand2"
            ).pack(side="left", padx=(0, 6))
        except Exception:
            pass

    def _on_speed_change(self, v):
        spd = round(float(v), 1)
        self.playback_speed = max(0.5, min(4.0, spd))
        self.speed_val_lbl.config(text=f"{self.playback_speed:.1f}x")

    def _set_speed(self, spd):
        self.playback_speed = float(spd)
        self.speed_slider.set(spd)
        self.speed_val_lbl.config(text=f"{spd:.1f}x")

    def _clear_report_stats(self):
        if hasattr(self, "reporter"):
            with self.reporter._lock:
                self.reporter._stats["incidents"] = 0
                self.reporter._stats["queued"] = 0
                self.reporter._stats["inventory"] = 0
                self.reporter._stats["dropped"] = 0
            if hasattr(self, "report_panel"):
                self.report_panel.last.set("Stats cleared. Ready for new run.")
                self.report_panel._refresh_status()

    def _report_log(self, msg):
        self.log_listbox.insert(0, msg)
        if self.log_listbox.size() > 35:
            self.log_listbox.delete(tk.END)

    def _on_close(self):
        self.stop_playback()
        if hasattr(self, "reporter"):
            self.reporter.shutdown()
        self.root.destroy()

    def _create_metric_row(self, parent, label, default_val):
        row = tk.Frame(parent, bg="#232634")
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text=label, font=("Segoe UI", 9), bg="#232634", fg="#a0aec0").pack(side=tk.LEFT)
        val_lbl = tk.Label(row, text=default_val, font=("Segoe UI", 9, "bold"), bg="#232634", fg="#ffffff")
        val_lbl.pack(side=tk.RIGHT)
        return val_lbl

    def _on_preset_change(self, event):
        preset_name = self.preset_var.get()
        info = PRESETS.get(preset_name, {})
        self.selected_video = info.get("video", "")
        self.selected_model = info.get("model", "")
        self.selected_model_type = info.get("model_type", "yolo")
        self.mode = info.get("mode", "general")
        self.lbl_module_desc.config(text=info.get("desc", ""))

        if self.mode == "track_defects":
            self.engine_badge.config(text="ENGINE: Faster R-CNN ResNet-50 FPN (Track Defects)", fg="#ffb74d")
            self.lbl_metric_name.config(text="Defect Severity:")
            self.lbl_metric_val.config(text="Nominal", fg="#48bb78")
            self.conf_slider.set(0.20)
            self.conf_val_lbl.config(text="0.20")
        elif self.mode == "distance_tracking":
            self.engine_badge.config(text="ENGINE: YOLOv11 (Obstacle Detection & Distance Engine)", fg="#00e5ff")
            self.lbl_metric_name.config(text="Track Corridor Clearance:")
            self.lbl_metric_val.config(text="Clear", fg="#48bb78")
            self.conf_slider.set(0.25)
            self.conf_val_lbl.config(text="0.25")
        else:
            self.engine_badge.config(text="ENGINE: YOLOv11 (Bridge Structural Defects)", fg="#ba68c8")
            self.lbl_metric_name.config(text="Bridge Condition:")
            self.lbl_metric_val.config(text="Clear", fg="#48bb78")
            self.conf_slider.set(0.25)
            self.conf_val_lbl.config(text="0.25")

        self.log_listbox.delete(0, tk.END)
        self.log_listbox.insert(tk.END, f"[PIPELINE] {preset_name}")
        self.log_listbox.insert(tk.END, f"[MODEL] {os.path.basename(self.selected_model)}")
        self.log_listbox.insert(tk.END, f"[SOURCE] {os.path.basename(self.selected_video)}")

    def _choose_custom_video(self):
        f = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if f:
            self.selected_video = f
            self.log_listbox.insert(tk.END, f"[SOURCE] Custom file: {os.path.basename(f)}")

    def start_playback(self):
        if not self.selected_video or not os.path.exists(self.selected_video):
            messagebox.showerror("Error", f"Video file not found:\n{self.selected_video}")
            return
        if not self.selected_model or not os.path.exists(self.selected_model):
            messagebox.showerror("Error", f"Model weights not found:\n{self.selected_model}")
            return

        # Clear stats from previous run so each run starts fresh with 0 incidents and 0 queued
        self._clear_report_stats()

        self.btn_start.config(state=tk.DISABLED)
        self.btn_pause.config(state=tk.NORMAL, text="⏸ PAUSE")
        self.btn_stop.config(state=tk.NORMAL)
        self.status_badge.config(text="STATUS: INSPECTING", bg="#2b6cb0", fg="#90cdf4")

        self.is_running = True
        self.is_paused = False

        thread = threading.Thread(target=self._process_video_thread, daemon=True)
        thread.start()

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.btn_pause.config(text="▶ RESUME", bg="#2b6cb0")
            self.status_badge.config(text="STATUS: PAUSED", bg="#744210", fg="#fbd38d")
        else:
            self.btn_pause.config(text="⏸ PAUSE", bg="#4a5568")
            self.status_badge.config(text="STATUS: INSPECTING", bg="#2b6cb0", fg="#90cdf4")

    def stop_playback(self):
        if hasattr(self, "reporter") and getattr(self, "_report_token", 0):
            self.reporter.end_session(self._report_token)
            self.reporter.generate_digest()
            self._report_token = 0
        self.is_running = False
        self.is_paused = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_pause.config(state=tk.DISABLED, text="⏸ PAUSE")
        self.btn_stop.config(state=tk.DISABLED)
        self.status_badge.config(text="STATUS: STOPPED", bg="#2d3748", fg="#e2e8f0")
        self.alert_label.config(
            text="⏹ Inspection Stopped",
            bg="#2d3748",
            fg="#e2e8f0"
        )

    def _process_video_thread(self):
        # Load appropriate model
        if self.current_model_path != self.selected_model:
            self.log_listbox.insert(tk.END, f"[LOAD] Initializing {self.selected_model_type.upper()} neural network...")
            if self.selected_model_type == "rcnn":
                self.current_model = load_rcnn_model(self.selected_model, num_classes=7)
            else:
                self.current_model = YOLO(self.selected_model)
            self.current_model_path = self.selected_model
            self.current_model_type = self.selected_model_type

        cap = open_capture(getattr(self, "capture_kind", "file"), self.selected_video)
        is_live = getattr(cap, "is_live", False)
        if not cap.isOpened():
            self.log_listbox.insert(tk.END, "[ERROR] Unable to open video stream.")
            self.stop_playback()
            return

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Adaptive focal length scaled to video resolution
        scaled_focal_length = BASE_FOCAL_LENGTH_640 * (max(frame_w, 640) / 640.0)

        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        preset_info = PRESETS.get(self.preset_var.get(), {})
        report_mod = preset_info.get("report_module")
        reporting_active = (report_mod is not None)
        report_token = 0
        if reporting_active:
            structure_id = os.path.splitext(os.path.basename(self.selected_video))[0] if report_mod == "bridge" else None
            source_type = "file" if not is_live else getattr(cap, "mode", "http_listen")
            video_label = cap.display_name() if is_live and hasattr(cap, "display_name") else self.selected_video
            report_token = self.reporter.start_session(
                video_label, self.current_model_path, self.current_model_type,
                fps, total_frames, float(self.conf_slider.get()),
                module=report_mod, structure_id=structure_id,
                is_live=is_live, source_type=source_type)
            self._report_token = report_token

        frame_idx = 0
        self.fps_timer = time.time()
        self.fps_count = 0

        while self.is_running and cap.isOpened():
            if self.is_paused:
                time.sleep(0.05)
                continue

            frame_start = time.time()
            if not is_live:
                speed_val = getattr(self, "playback_speed", 1.0)
                if speed_val > 1.2:
                    skip_count = int(round(speed_val)) - 1
                    for _ in range(skip_count):
                        if not cap.grab():
                            break
                        frame_idx += 1

            ret, frame = cap.read()
            if not ret:
                if is_live:
                    state_str = getattr(cap, "state", "RECONNECTING")
                    self.root.after(0, lambda s=state_str: self.alert_label.config(
                        text=f"📡 Stream {s.lower()} - waiting for frames...",
                        bg="#744210",
                        fg="#fbd38d"
                    ))
                    time.sleep(0.1)
                    continue
                else:
                    # Video playback complete
                    if reporting_active:
                        self.reporter.end_session(report_token)
                        self.reporter.generate_digest()
                    self.log_listbox.insert(tk.END, "[INFO] Video playback complete.")
                    break

            if is_live and (frame_w == 0 or frame_h == 0):
                frame_h, frame_w = frame.shape[:2]
                scaled_focal_length = BASE_FOCAL_LENGTH_640 * (max(frame_w, 640) / 640.0)

            raw_frame = frame.copy() if (reporting_active and self.current_model_type == "rcnn") else None
            report_dets = []

            frame_idx += 1
            conf_thresh = float(self.conf_slider.get())

            frame_alert = False
            has_lateral_feature = False
            closest_in_track_dist = float("inf")
            closest_lateral_dist = float("inf")
            detected_items = []

            # -------------------------------------------------------------
            # PIPELINE A: Track Defect Detection via Faster R-CNN
            # -------------------------------------------------------------
            if self.current_model_type == "rcnn":
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_tensor = F.to_tensor(img_rgb).unsqueeze(0)
                with torch.no_grad():
                    output = self.current_model(img_tensor)[0]

                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()

                for box, score, label_id in zip(boxes, scores, labels):
                    if score < conf_thresh:
                        continue
                    x1, y1, x2, y2 = map(int, box)
                    class_name = TRACK_CLASSES.get(int(label_id), f"Defect_{label_id}")

                    # Defect color palette
                    if class_name in ("Crack", "Squat", "Spalling"):
                        box_color = (0, 0, 255)       # Red for critical structural faults
                        frame_alert = True
                    elif class_name in ("Flaking", "Groove"):
                        box_color = (0, 165, 255)     # Orange
                    else:
                        box_color = (0, 255, 255)     # Yellow (Joints)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    label_str = f"{class_name} {score*100:.0f}%"
                    (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), box_color, -1)
                    cv2.putText(frame, label_str, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    detected_items.append(f"{class_name} ({score*100:.0f}%)")
                    report_dets.append((class_name, float(score), (x1, y1, x2, y2)))

            # -------------------------------------------------------------
            # PIPELINE B: Obstacle / Clearance / Bridge Detection via YOLOv11
            # -------------------------------------------------------------
            else:
                results = self.current_model(frame, verbose=False, conf=conf_thresh)
                if reporting_active:
                    for r in results:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = self.current_model.names.get(cls_id, f"ID_{cls_id}")
                            report_dets.append((class_name, conf, (x1, y1, x2, y2)))
                    raw_frame = frame.copy() if report_dets else None
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = self.current_model.names.get(cls_id, f"ID_{cls_id}")

                        if self.mode == "distance_tracking":
                            width_px = max(1, x2 - x1)
                            height_px = max(1, y2 - y1)
                            real_w = KNOWN_WIDTH.get(class_name, 60)
                            dist_cm = calculate_distance(scaled_focal_length, real_w, width_px)
                            dist_m = dist_cm / 100.0

                            # Lateral position analysis
                            cx = (x1 + x2) / 2.0
                            norm_cx = cx / float(frame_w)
                            norm_w = width_px / float(frame_w)
                            norm_y2 = y2 / float(frame_h)

                            # Check if the object is in the central Railway Track Corridor
                            # Central corridor is ~32% to 68% of horizontal field of view
                            is_central_corridor = (0.32 <= norm_cx <= 0.68)
                            
                            # Check if it's a huge tunnel entrance / portal / wall spanning peripheral sides
                            is_tunnel_or_wall = (norm_w > 0.40 and (x1 <= 50 or x2 >= frame_w - 50)) or not is_central_corridor

                            if is_tunnel_or_wall or not is_central_corridor:
                                # Lateral Clearance / Tunnel Proximity -> YELLOW tag (Safe Margin)
                                has_lateral_feature = True
                                box_color = (0, 215, 255)  # Amber / Yellow
                                label_str = f"{class_name} {conf*100:.0f}% | {dist_m:.1f}m (Clearance)"
                                detected_items.append(f"{class_name} ({dist_m:.1f}m, Side/Tunnel)")
                                if dist_m < closest_lateral_dist:
                                    closest_lateral_dist = dist_m
                            else:
                                # Central In-Track Obstacle
                                if dist_m < (WARNING_DISTANCE_CM / 100.0):
                                    # RED: Critical in-track obstacle within braking distance (<15m)
                                    box_color = (0, 0, 255)  # Red
                                    frame_alert = True
                                    label_str = f"{class_name} {conf*100:.0f}% | {dist_m:.1f}m [IN-TRACK ALERT]"
                                elif dist_m < 35.0:
                                    # AMBER: In-track obstacle approaching
                                    box_color = (0, 165, 255)  # Orange/Amber
                                    label_str = f"{class_name} {conf*100:.0f}% | {dist_m:.1f}m [APPROACHING]"
                                else:
                                    # GREEN: Safe in-track distance
                                    box_color = (0, 255, 0)  # Green
                                    label_str = f"{class_name} {conf*100:.0f}% | {dist_m:.1f}m [SAFE]"

                                detected_items.append(f"{class_name} ({dist_m:.1f}m, Track)")
                                if dist_m < closest_in_track_dist:
                                    closest_in_track_dist = dist_m
                        else:
                            # Bridge defect mode
                            box_color = (0, 0, 255) if class_name in ("crack", "spalling") else (0, 165, 255)
                            label_str = f"{class_name} {conf*100:.0f}%"
                            detected_items.append(f"{class_name} ({conf*100:.0f}%)")

                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                        (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), box_color, -1)
                        cv2.putText(frame, label_str, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # FPS calculation
            self.fps_count += 1
            elapsed = time.time() - self.fps_timer
            if elapsed >= 0.5:
                self.current_fps = round(self.fps_count / elapsed, 1)
                self.fps_count = 0
                self.fps_timer = time.time()

            # Schedule UI Update on Main Tkinter Thread
            self.root.after(
                0,
                self._update_ui_frame,
                frame.copy(),
                frame_idx,
                total_frames,
                len(detected_items),
                closest_in_track_dist,
                closest_lateral_dist,
                frame_alert,
                has_lateral_feature,
                detected_items
            )

            if reporting_active:
                self.reporter.ingest(report_token, frame_idx, raw_frame if report_dets else None, report_dets)

            if not is_live:
                # Dynamic sleep based on target FPS and playback speed
                speed_val = getattr(self, "playback_speed", 1.0)
                target_frame_time = (1.0 / (fps * speed_val)) if fps > 0 else 0.03
                loop_elapsed = time.time() - frame_start
                sleep_duration = max(0.001, target_frame_time - loop_elapsed)
                time.sleep(sleep_duration)

        if reporting_active:
            self.reporter.end_session(report_token)
            self.reporter.generate_digest()
        cap.release()
        self.root.after(0, self.stop_playback)

    def _update_ui_frame(self, frame_bgr, frame_idx, total_frames, obj_count, in_track_dist, lateral_dist, alert, lateral_feature, items):
        # Update metrics
        self.lbl_fps.config(text=f"{self.current_fps} FPS")
        if total_frames > 0:
            self.lbl_frame.config(text=f"{frame_idx} / {total_frames}")
        else:
            self.lbl_frame.config(text=f"LIVE #{frame_idx}")
        self.lbl_objects.config(text=str(obj_count))

        if self.mode == "distance_tracking":
            if in_track_dist < float("inf"):
                self.lbl_metric_name.config(text="Track Corridor Obstacle:")
                self.lbl_metric_val.config(text=f"{in_track_dist:.1f} m", fg="#ff4d4d" if alert else "#ffb74d")
            elif lateral_dist < float("inf"):
                self.lbl_metric_name.config(text="Tunnel / Lateral Clearance:")
                self.lbl_metric_val.config(text=f"{lateral_dist:.1f} m (Safe)", fg="#00e5ff")
            else:
                self.lbl_metric_name.config(text="Track Corridor Clearance:")
                self.lbl_metric_val.config(text="Clear Rail (Safe)", fg="#48bb78")
        elif self.mode == "track_defects":
            if alert:
                self.lbl_metric_val.config(text="DEFECT DETECTED", fg="#ff4d4d")
            elif obj_count > 0:
                self.lbl_metric_val.config(text="Minor Wear / Joint", fg="#ffb74d")
            else:
                self.lbl_metric_val.config(text="Clear Rail", fg="#48bb78")
        else:
            if obj_count > 0:
                self.lbl_metric_val.config(text=f"{obj_count} Anomaly", fg="#ffb74d")
            else:
                self.lbl_metric_val.config(text="Sound Structure", fg="#48bb78")

        # Update Alert Banner
        if self.mode == "distance_tracking":
            if alert:
                self.alert_label.config(
                    text="⚠️ CRITICAL BRAKING ALERT: OBSTACLE DIRECTLY ON TRACK (< 15m)!",
                    bg="#742a2a",
                    fg="#feb2b2"
                )
            elif lateral_feature:
                self.alert_label.config(
                    text=f"ℹ️ Tunnel Portal / Lateral Clearance Maintained (Safe Corridor)",
                    bg="#2d3748",
                    fg="#ffd700"
                )
            elif obj_count > 0:
                self.alert_label.config(
                    text=f"ℹ️ Track Corridor Feature Detected - Safe Stopping Margin",
                    bg="#2a4365",
                    fg="#90cdf4"
                )
            else:
                self.alert_label.config(
                    text="✅ Track Clear & Sound - Normal Train Operations",
                    bg="#1c2d27",
                    fg="#48bb78"
                )
        elif self.mode == "track_defects":
            if alert:
                self.alert_label.config(
                    text=f"⚠️ DEFECT DETECTED: {', '.join(items[:2])}",
                    bg="#742a2a",
                    fg="#feb2b2"
                )
            elif obj_count > 0:
                self.alert_label.config(
                    text=f"ℹ️ Track Observation: {', '.join(items[:2])}",
                    bg="#2a4365",
                    fg="#90cdf4"
                )
            else:
                self.alert_label.config(
                    text="✅ Rail Surface Nominal - No Faults Detected",
                    bg="#1c2d27",
                    fg="#48bb78"
                )
        else:
            if alert or obj_count > 0:
                self.alert_label.config(
                    text=f"⚠️ Bridge Anomaly: {', '.join(items[:2])}",
                    bg="#742a2a",
                    fg="#feb2b2"
                )
            else:
                self.alert_label.config(
                    text="✅ Bridge Structure Sound",
                    bg="#1c2d27",
                    fg="#48bb78"
                )

        # Log entries
        if items and frame_idx % 5 == 0:
            msg = f"F#{frame_idx}: " + ", ".join(items[:2])
            self.log_listbox.insert(0, msg)
            if self.log_listbox.size() > 35:
                self.log_listbox.delete(tk.END)

        # Canvas rendering
        canvas_w = self.video_canvas.winfo_width()
        canvas_h = self.video_canvas.winfo_height()

        if canvas_w > 80 and canvas_h > 80:
            h, w, _ = frame_bgr.shape
            scale = min(canvas_w / w, canvas_h / h)
            nw, nh = int(w * scale), int(h * scale)
            resized = cv2.resize(frame_bgr, (nw, nh))

            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            self.photo = ImageTk.PhotoImage(image=img)

            self.video_canvas.delete("all")
            self.video_canvas.create_image(
                canvas_w // 2, canvas_h // 2,
                image=self.photo,
                anchor=tk.CENTER
            )


def main():
    print("[INFO] Initializing Railway Infrastructure Inspection GUI dashboard...", flush=True)
    root = tk.Tk()
    app = RailwayInspectorApp(root)
    root.deiconify()
    root.lift()
    root.focus_force()
    root.attributes('-topmost', True)
    root.after(1000, lambda: root.attributes('-topmost', False))
    print("[INFO] Dashboard active and ready! Running main loop...", flush=True)
    root.mainloop()


if __name__ == "__main__":
    main()
