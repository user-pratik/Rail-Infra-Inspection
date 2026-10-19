"""
Railway Infrastructure Inspection System - Interactive GUI Dashboard
Features:
 1. Modern Dark Restyle (CustomTkinter) with Segmented Tabview:
    - Tab 1: "📹 Video Demo" (Presets 1-4, Custom video, speed control, telemetry, reporting)
    - Tab 2: "📡 Live Stream" (HTTP Stream Capture via FFmpeg listen/connect, live stats, 4 presets, reporting)
 2. Send Email Dispatch:
    - Daily Digest + Critical Incident PDFs + CSV export distribution via smtplib
 3. Unified Inference & Inspection Engine:
    - Obstacle & Clearance Engine (YOLOv11)
    - Track Defect Inspection (Faster R-CNN)
    - Bridge Structural Defect Inspection (YOLOv11)
"""

from __future__ import annotations

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
import customtkinter as ctk

from reporting.hub import ReportingHub
from reporting.gui_panel import ReportingPanel
from streaming import open_capture, HttpStreamCapture

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Theme Palette Constants
COLOR_BG_DARK = "#1a1b26"
COLOR_CARD_DARK = "#24283b"
COLOR_CARD_HEADER = "#1f2335"
COLOR_ACCENT = "#7aa2f7"
COLOR_ACCENT_HOVER = "#89b4fa"
COLOR_SUCCESS = "#9ece6a"
COLOR_WARNING = "#e0af68"
COLOR_DANGER = "#f7768e"
COLOR_TEXT = "#c0caf5"
COLOR_SUBTEXT = "#787c99"
COLOR_CANVAS_BG = "#0f0f14"

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
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title("Railway Infrastructure & Defect Inspection Suite")
        
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        win_w = min(1360, max(1080, screen_w - 40))
        win_h = min(920, max(720, screen_h - 60))
        pos_x = max(0, (screen_w - win_w) // 2)
        pos_y = max(0, (screen_h - win_h) // 2)
        self.root.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
        self.root.minsize(1040, 700)
        self.root.configure(fg_color=COLOR_BG_DARK)

        # Core State
        self.current_model = None
        self.current_model_path = ""
        self.current_model_type = ""
        self.cap = None
        self.is_running = False
        self.is_paused = False
        self.current_session_tab = "demo"  # "demo" or "live"
        self.capture_kind = "file"

        self.fps_timer = time.time()
        self.fps_count = 0
        self.current_fps = 0.0

        self.photo = None
        self.reporter = ReportingHub()
        self._report_token = 0
        self.playback_speed = 1.0

        # Build UI
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Initial trigger
        self._on_preset_change_demo(None)
        self._on_preset_change_live(None)

    def _build_ui(self):
        # 1. Header Bar
        header_frame = ctk.CTkFrame(self.root, fg_color="#13141c", height=65, corner_radius=0)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)

        title_box = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_box.pack(side=tk.LEFT, padx=20, pady=8)

        ctk.CTkLabel(
            title_box,
            text="🚄 RAILWAY INFRASTRUCTURE INSPECTION SUITE",
            font=("Segoe UI", 16, "bold"),
            text_color="#00e5ff"
        ).pack(anchor=tk.W)

        self.engine_badge = ctk.CTkLabel(
            title_box,
            text="ENGINE: YOLOv11 (Obstacle & Clearance Engine)",
            font=("Segoe UI", 10, "bold"),
            text_color=COLOR_SUBTEXT
        )
        self.engine_badge.pack(anchor=tk.W)

        self.status_badge = ctk.CTkLabel(
            header_frame,
            text="STATUS: READY",
            font=("Segoe UI", 11, "bold"),
            fg_color="#24283b",
            text_color=COLOR_TEXT,
            corner_radius=8,
            padx=14,
            pady=6
        )
        self.status_badge.pack(side=tk.RIGHT, padx=20, pady=14)

        # 2. Main Tabview
        self.tabview = ctk.CTkTabview(
            self.root,
            fg_color=COLOR_BG_DARK,
            segmented_button_fg_color="#1f2335",
            segmented_button_selected_color=COLOR_ACCENT,
            segmented_button_selected_hover_color=COLOR_ACCENT_HOVER,
            segmented_button_unselected_color="#24283b",
            segmented_button_unselected_hover_color="#2e344e",
            text_color="white",
            corner_radius=10
        )
        self.tabview.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))

        self.tab_demo = self.tabview.add("📹 Video Demo")
        self.tab_live = self.tabview.add("📡 Live Stream")

        self._build_demo_tab(self.tab_demo)
        self._build_live_tab(self.tab_live)

    # =========================================================================
    # TAB 1: VIDEO DEMO
    # =========================================================================
    def _build_demo_tab(self, parent):
        main_frame = ctk.CTkFrame(parent, fg_color="transparent")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Left Column: Video Screen + Live Alert Banner
        left_col = ctk.CTkFrame(main_frame, fg_color="transparent")
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.video_canvas_demo = tk.Canvas(
            left_col,
            bg=COLOR_CANVAS_BG,
            highlightthickness=2,
            highlightbackground="#24283b"
        )
        self.video_canvas_demo.pack(fill=tk.BOTH, expand=True)

        self.alert_label_demo = tk.Label(
            left_col,
            text="✅ System Initialized - Ready to start video demo inspection",
            font=("Segoe UI", 11, "bold"),
            bg="#1c2d27",
            fg=COLOR_SUCCESS,
            pady=8
        )
        self.alert_label_demo.pack(fill=tk.X, pady=(6, 0))

        # Right Column: Controls & Telemetry
        right_col = ctk.CTkScrollableFrame(main_frame, fg_color="transparent", width=380)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(0, 0))

        # 1. Preset Selector Card
        preset_card = tk.LabelFrame(
            right_col,
            text=" ⚙️ Inspection Module ",
            font=("Segoe UI", 10, "bold"),
            bg=COLOR_CARD_DARK,
            fg=COLOR_TEXT,
            padx=12,
            pady=8
        )
        preset_card.pack(fill=tk.X, pady=(0, 8))

        tk.Label(preset_card, text="Select Inspection Pipeline:", font=("Segoe UI", 9), bg=COLOR_CARD_DARK, fg=COLOR_SUBTEXT).pack(anchor=tk.W)
        self.preset_var_demo = tk.StringVar(value=list(PRESETS.keys())[0])
        self.dropdown_demo = ttk.Combobox(
            preset_card,
            textvariable=self.preset_var_demo,
            values=list(PRESETS.keys()),
            state="readonly",
            font=("Segoe UI", 9)
        )
        self.dropdown_demo.pack(fill=tk.X, pady=(4, 6))
        self.dropdown_demo.bind("<<ComboboxSelected>>", self._on_preset_change_demo)

        self.lbl_module_desc_demo = tk.Label(
            preset_card,
            text="",
            font=("Segoe UI", 8, "italic"),
            bg=COLOR_CARD_DARK,
            fg=COLOR_SUBTEXT,
            wraplength=340,
            justify=tk.LEFT
        )
        self.lbl_module_desc_demo.pack(anchor=tk.W, pady=(0, 6))

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
        btn_custom.pack(fill=tk.X, pady=(0, 2))

        # 2. Controls Card
        ctrl_card = tk.LabelFrame(
            right_col,
            text=" 🎮 Playback & Inference Controls ",
            font=("Segoe UI", 10, "bold"),
            bg=COLOR_CARD_DARK,
            fg=COLOR_TEXT,
            padx=12,
            pady=8
        )
        ctrl_card.pack(fill=tk.X, pady=(0, 8))

        btn_row = tk.Frame(ctrl_card, bg=COLOR_CARD_DARK)
        btn_row.pack(fill=tk.X, pady=(0, 6))

        self.btn_start_demo = tk.Button(
            btn_row,
            text="▶ START",
            font=("Segoe UI", 10, "bold"),
            bg="#2b6cb0",
            fg="white",
            activebackground="#2c5282",
            relief=tk.FLAT,
            command=self.start_playback
        )
        self.btn_start_demo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))

        self.btn_pause_demo = tk.Button(
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
        self.btn_pause_demo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)

        self.btn_stop_demo = tk.Button(
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
        self.btn_stop_demo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))

        # Report & Digest Buttons in Playback Controls
        report_btn_row = tk.Frame(ctrl_card, bg=COLOR_CARD_DARK)
        report_btn_row.pack(fill=tk.X, pady=(0, 6))

        tk.Button(
            report_btn_row,
            text="📑 Daily Digest",
            font=("Segoe UI", 9, "bold"),
            bg="#2c5282",
            fg="white",
            activebackground="#2b6cb0",
            relief=tk.FLAT,
            padx=6,
            pady=3,
            cursor="hand2",
            command=lambda: self.reporter.generate_digest(auto_open=True)
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))

        tk.Button(
            report_btn_row,
            text="📄 Report",
            font=("Segoe UI", 9, "bold"),
            bg="#319795",
            fg="white",
            activebackground="#285e61",
            relief=tk.FLAT,
            padx=6,
            pady=3,
            cursor="hand2",
            command=lambda: self.reporter.compile_report(auto_open=True)
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))

        # Confidence Slider
        slider_row = tk.Frame(ctrl_card, bg=COLOR_CARD_DARK)
        slider_row.pack(fill=tk.X, pady=(2, 0))
        tk.Label(slider_row, text="Confidence Threshold:", font=("Segoe UI", 9), bg=COLOR_CARD_DARK, fg=COLOR_SUBTEXT).pack(side=tk.LEFT)
        self.conf_val_lbl_demo = tk.Label(slider_row, text="0.25", font=("Segoe UI", 9, "bold"), bg=COLOR_CARD_DARK, fg="#00e5ff")
        self.conf_val_lbl_demo.pack(side=tk.RIGHT)

        self.conf_slider_demo = ttk.Scale(
            ctrl_card,
            from_=0.10,
            to=0.95,
            value=0.25,
            command=self._on_conf_change_demo
        )
        self.conf_slider_demo.pack(fill=tk.X, pady=(2, 4))

        # Playback Speed Slider & Controls (0.5x to 4.0x)
        speed_row = tk.Frame(ctrl_card, bg=COLOR_CARD_DARK)
        speed_row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(speed_row, text="Playback Speed:", font=("Segoe UI", 9), bg=COLOR_CARD_DARK, fg=COLOR_SUBTEXT).pack(side=tk.LEFT)
        self.speed_val_lbl_demo = tk.Label(speed_row, text="1.0x", font=("Segoe UI", 9, "bold"), bg=COLOR_CARD_DARK, fg="#00e5ff")
        self.speed_val_lbl_demo.pack(side=tk.RIGHT)

        self.speed_slider_demo = ttk.Scale(
            ctrl_card,
            from_=0.5,
            to=4.0,
            value=1.0,
            command=self._on_speed_change
        )
        self.speed_slider_demo.pack(fill=tk.X, pady=(2, 2))

        speed_btn_row = tk.Frame(ctrl_card, bg=COLOR_CARD_DARK)
        speed_btn_row.pack(fill=tk.X, pady=(0, 2))
        for spd in [0.5, 1.0, 2.0, 3.0, 4.0]:
            tk.Button(
                speed_btn_row,
                text=f"{spd}x",
                font=("Segoe UI", 8),
                bg="#3b4252",
                fg="#eceff4",
                activebackground="#4c566a",
                relief=tk.FLAT,
                command=lambda s=spd: self._set_speed(s)
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)

        # 3. Telemetry & Metrics Card
        metrics_card = tk.LabelFrame(
            right_col,
            text=" 📊 Real-Time Telemetry & Insights ",
            font=("Segoe UI", 10, "bold"),
            bg=COLOR_CARD_DARK,
            fg=COLOR_TEXT,
            padx=12,
            pady=8
        )
        metrics_card.pack(fill=tk.X, pady=(0, 8))

        self.lbl_fps_demo = self._create_metric_row(metrics_card, "Processing FPS:", "0.0 FPS")
        self.lbl_frame_demo = self._create_metric_row(metrics_card, "Frame Counter:", "0 / 0")
        self.lbl_objects_demo = self._create_metric_row(metrics_card, "Detections in Frame:", "0")

        self.metric_row_frame_demo = tk.Frame(metrics_card, bg=COLOR_CARD_DARK)
        self.metric_row_frame_demo.pack(fill=tk.X, pady=2)
        self.lbl_metric_name_demo = tk.Label(self.metric_row_frame_demo, text="Key Finding:", font=("Segoe UI", 9), bg=COLOR_CARD_DARK, fg=COLOR_SUBTEXT)
        self.lbl_metric_name_demo.pack(side=tk.LEFT)
        self.lbl_metric_val_demo = tk.Label(self.metric_row_frame_demo, text="Normal", font=("Segoe UI", 9, "bold"), bg=COLOR_CARD_DARK, fg="#ffffff")
        self.lbl_metric_val_demo.pack(side=tk.RIGHT)

        tk.Label(metrics_card, text="Live Inspection Log:", font=("Segoe UI", 9, "bold"), bg=COLOR_CARD_DARK, fg=COLOR_SUBTEXT).pack(anchor=tk.W, pady=(4, 2))
        self.log_listbox_demo = tk.Listbox(
            metrics_card,
            height=4,
            bg="#151720",
            fg=COLOR_TEXT,
            font=("Consolas", 8),
            highlightthickness=0,
            selectbackground="#2b6cb0"
        )
        self.log_listbox_demo.pack(fill=tk.X, pady=(0, 2))

        # 4. Reporting Panel
        self.report_panel_demo = ReportingPanel(
            right_col,
            self.reporter,
            self.root,
            log_fn=self._report_log_demo,
            bg=COLOR_CARD_DARK,
            fg=COLOR_TEXT,
            accent="#2b6cb0",
            font=("Segoe UI", 9)
        )
        self.report_panel_demo.pack(fill=tk.X, pady=(0, 4))

        # Clear Button
        try:
            panel_btn_row = self.report_panel_demo.frame.winfo_children()[-1]
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

    # =========================================================================
    # TAB 2: LIVE STREAM
    # =========================================================================
    def _build_live_tab(self, parent):
        main_frame = ctk.CTkFrame(parent, fg_color="transparent")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Left Column: Video Screen + Live Alert Banner
        left_col = ctk.CTkFrame(main_frame, fg_color="transparent")
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.video_canvas_live = tk.Canvas(
            left_col,
            bg=COLOR_CANVAS_BG,
            highlightthickness=2,
            highlightbackground="#24283b"
        )
        self.video_canvas_live.pack(fill=tk.BOTH, expand=True)

        self.alert_label_live = tk.Label(
            left_col,
            text="📡 Live Stream Inactive - Click START LIVE STREAM to begin ingestion",
            font=("Segoe UI", 11, "bold"),
            bg="#1c2d27",
            fg=COLOR_SUCCESS,
            pady=8
        )
        self.alert_label_live.pack(fill=tk.X, pady=(6, 0))

        # Right Column: Controls & Telemetry
        right_col = ctk.CTkScrollableFrame(main_frame, fg_color="transparent", width=380)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(0, 0))

        # 1. Stream Configuration Card
        stream_cfg_card = tk.LabelFrame(
            right_col,
            text=" 📡 HTTP Live Stream Configuration ",
            font=("Segoe UI", 10, "bold"),
            bg=COLOR_CARD_DARK,
            fg=COLOR_TEXT,
            padx=12,
            pady=8
        )
        stream_cfg_card.pack(fill=tk.X, pady=(0, 8))

        # Mode selector
        mode_row = tk.Frame(stream_cfg_card, bg=COLOR_CARD_DARK)
        mode_row.pack(fill=tk.X, pady=(0, 4))
        tk.Label(mode_row, text="Stream Mode:", font=("Segoe UI", 9, "bold"), bg=COLOR_CARD_DARK, fg=COLOR_SUBTEXT).pack(side=tk.LEFT)
        self.live_mode_var = tk.StringVar(value="listen")
        self.mode_combo = ttk.Combobox(
            mode_row,
            textvariable=self.live_mode_var,
            values=["listen (Hardware POSTs to PC)", "connect (PC pulls from Camera)"],
            state="readonly",
            font=("Segoe UI", 8),
            width=28
        )
        self.mode_combo.pack(side=tk.RIGHT)

        # Host / Bind & Port row
        host_row = tk.Frame(stream_cfg_card, bg=COLOR_CARD_DARK)
        host_row.pack(fill=tk.X, pady=2)
        tk.Label(host_row, text="Host/Bind:", font=("Segoe UI", 9), bg=COLOR_CARD_DARK, fg=COLOR_SUBTEXT).pack(side=tk.LEFT)
        self.live_host_entry = tk.Entry(host_row, font=("Segoe UI", 9), bg="#151720", fg="white", width=14, insertbackground="white")
        self.live_host_entry.insert(0, "0.0.0.0")
        self.live_host_entry.pack(side=tk.LEFT, padx=4)

        tk.Label(host_row, text="Port:", font=("Segoe UI", 9), bg=COLOR_CARD_DARK, fg=COLOR_SUBTEXT).pack(side=tk.LEFT, padx=(4, 2))
        self.live_port_entry = tk.Entry(host_row, font=("Segoe UI", 9), bg="#151720", fg="white", width=6, insertbackground="white")
        self.live_port_entry.insert(0, "8554")
        self.live_port_entry.pack(side=tk.LEFT)

        path_row = tk.Frame(stream_cfg_card, bg=COLOR_CARD_DARK)
        path_row.pack(fill=tk.X, pady=2)
        tk.Label(path_row, text="Path:", font=("Segoe UI", 9), bg=COLOR_CARD_DARK, fg=COLOR_SUBTEXT).pack(side=tk.LEFT)
        self.live_path_entry = tk.Entry(path_row, font=("Segoe UI", 9), bg="#151720", fg="white", insertbackground="white")
        self.live_path_entry.insert(0, "/live")
        self.live_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        # Connection State & Ingest Stats Pill
        pill_row = tk.Frame(stream_cfg_card, bg=COLOR_CARD_DARK)
        pill_row.pack(fill=tk.X, pady=(6, 2))
        self.live_state_badge = tk.Label(
            pill_row,
            text="DISCONNECTED",
            font=("Segoe UI", 8, "bold"),
            bg="#414868",
            fg="#c0caf5",
            padx=8,
            pady=2
        )
        self.live_state_badge.pack(side=tk.LEFT)

        self.lbl_live_ingest_stats = tk.Label(
            pill_row,
            text="Ingest: 0.0 FPS | Drops: 0",
            font=("Segoe UI", 8),
            bg=COLOR_CARD_DARK,
            fg=COLOR_SUBTEXT
        )
        self.lbl_live_ingest_stats.pack(side=tk.RIGHT)

        tk.Label(
            stream_cfg_card,
            text="ℹ️ Note: FFmpeg HTTP listener has no auth. Use private LAN or hotspot.",
            font=("Segoe UI", 7, "italic"),
            bg=COLOR_CARD_DARK,
            fg="#e0af68"
        ).pack(anchor=tk.W, pady=(4, 0))

        # 2. Pipeline Selector for Live Stream
        preset_card_live = tk.LabelFrame(
            right_col,
            text=" ⚙️ Live Inspection Pipeline ",
            font=("Segoe UI", 10, "bold"),
            bg=COLOR_CARD_DARK,
            fg=COLOR_TEXT,
            padx=12,
            pady=8
        )
        preset_card_live.pack(fill=tk.X, pady=(0, 8))

        tk.Label(preset_card_live, text="Select Model & Inspection Mode:", font=("Segoe UI", 9), bg=COLOR_CARD_DARK, fg=COLOR_SUBTEXT).pack(anchor=tk.W)
        self.preset_var_live = tk.StringVar(value=list(PRESETS.keys())[3])  # Default to Bridge inspection
        self.dropdown_live = ttk.Combobox(
            preset_card_live,
            textvariable=self.preset_var_live,
            values=list(PRESETS.keys()),
            state="readonly",
            font=("Segoe UI", 9)
        )
        self.dropdown_live.pack(fill=tk.X, pady=(4, 6))
        self.dropdown_live.bind("<<ComboboxSelected>>", self._on_preset_change_live)

        self.lbl_module_desc_live = tk.Label(
            preset_card_live,
            text="",
            font=("Segoe UI", 8, "italic"),
            bg=COLOR_CARD_DARK,
            fg=COLOR_SUBTEXT,
            wraplength=340,
            justify=tk.LEFT
        )
        self.lbl_module_desc_live.pack(anchor=tk.W, pady=(0, 2))

        # 3. Live Playback & Inference Controls
        ctrl_card_live = tk.LabelFrame(
            right_col,
            text=" 🎮 Live Inference Controls ",
            font=("Segoe UI", 10, "bold"),
            bg=COLOR_CARD_DARK,
            fg=COLOR_TEXT,
            padx=12,
            pady=8
        )
        ctrl_card_live.pack(fill=tk.X, pady=(0, 8))

        btn_row_live = tk.Frame(ctrl_card_live, bg=COLOR_CARD_DARK)
        btn_row_live.pack(fill=tk.X, pady=(0, 6))

        self.btn_start_live = tk.Button(
            btn_row_live,
            text="📡 START STREAM",
            font=("Segoe UI", 9, "bold"),
            bg="#2b6cb0",
            fg="white",
            activebackground="#2c5282",
            relief=tk.FLAT,
            command=self.start_live_playback
        )
        self.btn_start_live.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))

        self.btn_pause_live = tk.Button(
            btn_row_live,
            text="⏸ PAUSE",
            font=("Segoe UI", 9, "bold"),
            bg="#4a5568",
            fg="white",
            activebackground="#2d3748",
            state=tk.DISABLED,
            relief=tk.FLAT,
            command=self.toggle_pause
        )
        self.btn_pause_live.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)

        self.btn_stop_live = tk.Button(
            btn_row_live,
            text="⏹ STOP",
            font=("Segoe UI", 9, "bold"),
            bg="#9b2c2c",
            fg="white",
            activebackground="#742a2a",
            state=tk.DISABLED,
            relief=tk.FLAT,
            command=self.stop_playback
        )
        self.btn_stop_live.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))

        # Confidence Slider
        slider_row_live = tk.Frame(ctrl_card_live, bg=COLOR_CARD_DARK)
        slider_row_live.pack(fill=tk.X, pady=(2, 0))
        tk.Label(slider_row_live, text="Confidence Threshold:", font=("Segoe UI", 9), bg=COLOR_CARD_DARK, fg=COLOR_SUBTEXT).pack(side=tk.LEFT)
        self.conf_val_lbl_live = tk.Label(slider_row_live, text="0.25", font=("Segoe UI", 9, "bold"), bg=COLOR_CARD_DARK, fg="#00e5ff")
        self.conf_val_lbl_live.pack(side=tk.RIGHT)

        self.conf_slider_live = ttk.Scale(
            ctrl_card_live,
            from_=0.10,
            to=0.95,
            value=0.25,
            command=self._on_conf_change_live
        )
        self.conf_slider_live.pack(fill=tk.X, pady=(2, 4))

        # 4. Telemetry Card for Live Stream
        metrics_card_live = tk.LabelFrame(
            right_col,
            text=" 📊 Live Telemetry & Insights ",
            font=("Segoe UI", 10, "bold"),
            bg=COLOR_CARD_DARK,
            fg=COLOR_TEXT,
            padx=12,
            pady=8
        )
        metrics_card_live.pack(fill=tk.X, pady=(0, 8))

        self.lbl_fps_live = self._create_metric_row(metrics_card_live, "Inference FPS:", "0.0 FPS")
        self.lbl_frame_live = self._create_metric_row(metrics_card_live, "Frame Counter:", "LIVE #0")
        self.lbl_objects_live = self._create_metric_row(metrics_card_live, "Detections in Frame:", "0")

        self.metric_row_frame_live = tk.Frame(metrics_card_live, bg=COLOR_CARD_DARK)
        self.metric_row_frame_live.pack(fill=tk.X, pady=2)
        self.lbl_metric_name_live = tk.Label(self.metric_row_frame_live, text="Condition:", font=("Segoe UI", 9), bg=COLOR_CARD_DARK, fg=COLOR_SUBTEXT)
        self.lbl_metric_name_live.pack(side=tk.LEFT)
        self.lbl_metric_val_live = tk.Label(self.metric_row_frame_live, text="Ready", font=("Segoe UI", 9, "bold"), bg=COLOR_CARD_DARK, fg="#ffffff")
        self.lbl_metric_val_live.pack(side=tk.RIGHT)

        tk.Label(metrics_card_live, text="Live Inspection Log:", font=("Segoe UI", 9, "bold"), bg=COLOR_CARD_DARK, fg=COLOR_SUBTEXT).pack(anchor=tk.W, pady=(4, 2))
        self.log_listbox_live = tk.Listbox(
            metrics_card_live,
            height=4,
            bg="#151720",
            fg=COLOR_TEXT,
            font=("Consolas", 8),
            highlightthickness=0,
            selectbackground="#2b6cb0"
        )
        self.log_listbox_live.pack(fill=tk.X, pady=(0, 2))

        # 5. Reporting Panel for Live Stream
        self.report_panel_live = ReportingPanel(
            right_col,
            self.reporter,
            self.root,
            log_fn=self._report_log_live,
            bg=COLOR_CARD_DARK,
            fg=COLOR_TEXT,
            accent="#2b6cb0",
            font=("Segoe UI", 9)
        )
        self.report_panel_live.pack(fill=tk.X, pady=(0, 4))

        try:
            panel_btn_row = self.report_panel_live.frame.winfo_children()[-1]
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

    # =========================================================================
    # Helpers & Callbacks
    # =========================================================================
    def _create_metric_row(self, parent, label, default_val):
        row = tk.Frame(parent, bg=COLOR_CARD_DARK)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text=label, font=("Segoe UI", 9), bg=COLOR_CARD_DARK, fg=COLOR_SUBTEXT).pack(side=tk.LEFT)
        val_lbl = tk.Label(row, text=default_val, font=("Segoe UI", 9, "bold"), bg=COLOR_CARD_DARK, fg="#ffffff")
        val_lbl.pack(side=tk.RIGHT)
        return val_lbl

    def _on_speed_change(self, v):
        spd = round(float(v), 1)
        self.playback_speed = max(0.5, min(4.0, spd))
        self.speed_val_lbl_demo.config(text=f"{self.playback_speed:.1f}x")

    def _set_speed(self, spd):
        self.playback_speed = float(spd)
        self.speed_slider_demo.set(spd)
        self.speed_val_lbl_demo.config(text=f"{spd:.1f}x")

    def _clear_report_stats(self):
        if hasattr(self, "reporter"):
            with self.reporter._lock:
                self.reporter._stats["incidents"] = 0
                self.reporter._stats["queued"] = 0
                self.reporter._stats["inventory"] = 0
                self.reporter._stats["dropped"] = 0
            if hasattr(self, "report_panel_demo"):
                self.report_panel_demo.last.set("Stats cleared. Ready for new run.")
                self.report_panel_demo._refresh_status()
            if hasattr(self, "report_panel_live"):
                self.report_panel_live.last.set("Stats cleared. Ready for new run.")
                self.report_panel_live._refresh_status()

    def _report_log_demo(self, msg):
        self.log_listbox_demo.insert(0, msg)
        if self.log_listbox_demo.size() > 35:
            self.log_listbox_demo.delete(tk.END)

    def _report_log_live(self, msg):
        self.log_listbox_live.insert(0, msg)
        if self.log_listbox_live.size() > 35:
            self.log_listbox_live.delete(tk.END)

    def _choose_custom_video(self):
        f = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if f:
            self.selected_video = f
            self.log_listbox_demo.insert(tk.END, f"[SOURCE] Custom file: {os.path.basename(f)}")

    def _on_conf_change_demo(self, v):
        val = float(v)
        self.conf_val_lbl_demo.config(text=f"{val:.2f}")
        if self.current_session_tab == "demo":
            self.active_conf_thresh = val

    def _on_conf_change_live(self, v):
        val = float(v)
        self.conf_val_lbl_live.config(text=f"{val:.2f}")
        if self.current_session_tab == "live":
            self.active_conf_thresh = val

    def _on_preset_change_demo(self, event):
        preset_name = self.preset_var_demo.get()
        info = PRESETS.get(preset_name, {})
        self.selected_video = info.get("video", "")
        self.selected_model = info.get("model", "")
        self.selected_model_type = info.get("model_type", "yolo")
        self.mode = info.get("mode", "general")
        self.lbl_module_desc_demo.config(text=info.get("desc", ""))

        if self.mode == "track_defects":
            self.engine_badge.configure(text="ENGINE: Faster R-CNN ResNet-50 FPN (Track Defects)", text_color="#ffb74d")
            self.lbl_metric_name_demo.config(text="Defect Severity:")
            self.lbl_metric_val_demo.config(text="Nominal", fg=COLOR_SUCCESS)
            self.conf_slider_demo.set(0.20)
            self.conf_val_lbl_demo.config(text="0.20")
        elif self.mode == "distance_tracking":
            self.engine_badge.configure(text="ENGINE: YOLOv11 (Obstacle & Distance Engine)", text_color="#00e5ff")
            self.lbl_metric_name_demo.config(text="Corridor Clearance:")
            self.lbl_metric_val_demo.config(text="Clear", fg=COLOR_SUCCESS)
            self.conf_slider_demo.set(0.25)
            self.conf_val_lbl_demo.config(text="0.25")
        else:
            self.engine_badge.configure(text="ENGINE: YOLOv11 (Bridge Structural Defects)", text_color="#ba68c8")
            self.lbl_metric_name_demo.config(text="Bridge Condition:")
            self.lbl_metric_val_demo.config(text="Clear", fg=COLOR_SUCCESS)
            self.conf_slider_demo.set(0.25)
            self.conf_val_lbl_demo.config(text="0.25")

        self.log_listbox_demo.delete(0, tk.END)
        self.log_listbox_demo.insert(tk.END, f"[PIPELINE] {preset_name}")
        self.log_listbox_demo.insert(tk.END, f"[MODEL] {os.path.basename(self.selected_model)}")
        self.log_listbox_demo.insert(tk.END, f"[SOURCE] {os.path.basename(self.selected_video)}")

    def _on_preset_change_live(self, event):
        preset_name = self.preset_var_live.get()
        info = PRESETS.get(preset_name, {})
        self.selected_model_live = info.get("model", "")
        self.selected_model_type_live = info.get("model_type", "yolo")
        self.mode_live = info.get("mode", "general")
        self.lbl_module_desc_live.config(text=info.get("desc", ""))

        if self.mode_live == "track_defects":
            self.lbl_metric_name_live.config(text="Defect Severity:")
            self.lbl_metric_val_live.config(text="Nominal", fg=COLOR_SUCCESS)
            self.conf_slider_live.set(0.20)
            self.conf_val_lbl_live.config(text="0.20")
        elif self.mode_live == "distance_tracking":
            self.lbl_metric_name_live.config(text="Corridor Clearance:")
            self.lbl_metric_val_live.config(text="Clear", fg=COLOR_SUCCESS)
            self.conf_slider_live.set(0.25)
            self.conf_val_lbl_live.config(text="0.25")
        else:
            self.lbl_metric_name_live.config(text="Bridge Condition:")
            self.lbl_metric_val_live.config(text="Clear", fg=COLOR_SUCCESS)
            self.conf_slider_live.set(0.25)
            self.conf_val_lbl_live.config(text="0.25")

        self.log_listbox_live.delete(0, tk.END)
        self.log_listbox_live.insert(tk.END, f"[LIVE PIPELINE] {preset_name}")
        self.log_listbox_live.insert(tk.END, f"[MODEL] {os.path.basename(self.selected_model_live)}")

    # =========================================================================
    # Playback Controls (Mutual Exclusion between Demo & Live)
    # =========================================================================
    def start_playback(self):
        """Starts video demo playback on Tab 1."""
        if self.is_running:
            messagebox.showwarning("Session Active", "An inspection session is already running. Please stop it first.")
            return

        if not self.selected_video or not os.path.exists(self.selected_video):
            messagebox.showerror("Error", f"Video file not found:\n{self.selected_video}")
            return
        if not self.selected_model or not os.path.exists(self.selected_model):
            messagebox.showerror("Error", f"Model weights not found:\n{self.selected_model}")
            return

        self._clear_report_stats()
        self.current_session_tab = "demo"
        self.capture_kind = "file"
        self.active_preset_name = self.preset_var_demo.get()
        self.active_conf_thresh = float(self.conf_slider_demo.get())

        # Toggles & Mutual Exclusion
        self.btn_start_demo.config(state=tk.DISABLED)
        self.btn_pause_demo.config(state=tk.NORMAL, text="⏸ PAUSE")
        self.btn_stop_demo.config(state=tk.NORMAL)
        self.btn_start_live.config(state=tk.DISABLED)

        self.status_badge.configure(text="STATUS: DEMO INSPECTING", fg_color="#2b6cb0", text_color="#90cdf4")
        self.is_running = True
        self.is_paused = False

        thread = threading.Thread(target=self._process_video_thread, daemon=True)
        thread.start()

    def start_live_playback(self):
        """Starts live stream capture & inference on Tab 2."""
        if self.is_running:
            messagebox.showwarning("Session Active", "An inspection session is already running. Please stop it first.")
            return

        # Prepare parameters from live configuration
        mode_raw = self.live_mode_var.get()
        mode = "listen" if "listen" in mode_raw.lower() else "connect"
        host = self.live_host_entry.get().strip() or ("0.0.0.0" if mode == "listen" else "127.0.0.1")
        try:
            port = int(self.live_port_entry.get().strip() or "8554")
        except ValueError:
            messagebox.showerror("Invalid Port", "Port must be an integer (e.g. 8554).")
            return
        path = self.live_path_entry.get().strip() or "/live"

        self.selected_model = self.selected_model_live
        self.selected_model_type = self.selected_model_type_live
        self.mode = self.mode_live

        if not self.selected_model or not os.path.exists(self.selected_model):
            messagebox.showerror("Error", f"Model weights not found:\n{self.selected_model}")
            return

        self._clear_report_stats()
        self.current_session_tab = "live"
        self.capture_kind = mode
        self.active_preset_name = self.preset_var_live.get()
        self.active_conf_thresh = float(self.conf_slider_live.get())
        self.live_stream_kwargs = {
            "mode": mode,
            "host": host,
            "bind": host,
            "port": port,
            "path": path,
            "read_timeout": 5.0
        }

        # Toggles & Mutual Exclusion
        self.btn_start_live.config(state=tk.DISABLED)
        self.btn_pause_live.config(state=tk.NORMAL, text="⏸ PAUSE")
        self.btn_stop_live.config(state=tk.NORMAL)
        self.btn_start_demo.config(state=tk.DISABLED)

        self.status_badge.configure(text="STATUS: LIVE INSPECTING", fg_color="#134e4a", text_color="#48bb78")
        self.live_state_badge.config(text="CONNECTING", bg="#744210", fg="#fbd38d")

        self.is_running = True
        self.is_paused = False

        thread = threading.Thread(target=self._process_video_thread, daemon=True)
        thread.start()

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            if self.current_session_tab == "demo":
                self.btn_pause_demo.config(text="▶ RESUME", bg="#2b6cb0")
            else:
                self.btn_pause_live.config(text="▶ RESUME", bg="#2b6cb0")
            self.status_badge.configure(text="STATUS: PAUSED", fg_color="#744210", text_color="#fbd38d")
        else:
            if self.current_session_tab == "demo":
                self.btn_pause_demo.config(text="⏸ PAUSE", bg="#4a5568")
                self.status_badge.configure(text="STATUS: DEMO INSPECTING", fg_color="#2b6cb0", text_color="#90cdf4")
            else:
                self.btn_pause_live.config(text="⏸ PAUSE", bg="#4a5568")
                self.status_badge.configure(text="STATUS: LIVE INSPECTING", fg_color="#134e4a", text_color="#48bb78")

    def stop_playback(self):
        if hasattr(self, "reporter") and getattr(self, "_report_token", 0):
            self.reporter.end_session(self._report_token)
            self.reporter.generate_digest()
            self._report_token = 0

        self.is_running = False
        self.is_paused = False

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        # Re-enable controls on both tabs
        self.btn_start_demo.config(state=tk.NORMAL)
        self.btn_pause_demo.config(state=tk.DISABLED, text="⏸ PAUSE")
        self.btn_stop_demo.config(state=tk.DISABLED)

        self.btn_start_live.config(state=tk.NORMAL)
        self.btn_pause_live.config(state=tk.DISABLED, text="⏸ PAUSE")
        self.btn_stop_live.config(state=tk.DISABLED)

        self.status_badge.configure(text="STATUS: STOPPED", fg_color="#24283b", text_color=COLOR_TEXT)
        self.live_state_badge.config(text="DISCONNECTED", bg="#414868", fg="#c0caf5")

        self.alert_label_demo.config(text="⏹ Demo Inspection Stopped", bg="#24283b", fg=COLOR_TEXT)
        self.alert_label_live.config(text="⏹ Live Stream Stopped", bg="#24283b", fg=COLOR_TEXT)

    def _safe_log(self, listbox, msg: str):
        try:
            self.root.after(0, lambda: (listbox.insert(tk.END, msg), listbox.see(tk.END)))
        except Exception:
            pass

    # =========================================================================
    # Unified Inference & Processing Thread
    # =========================================================================
    def _process_video_thread(self):
        is_live = (self.current_session_tab == "live")
        active_listbox = self.log_listbox_live if is_live else self.log_listbox_demo
        active_slider = self.conf_slider_live if is_live else self.conf_slider_demo

        # Load appropriate model
        if self.current_model_path != self.selected_model:
            self._safe_log(active_listbox, f"[LOAD] Initializing {self.selected_model_type.upper()} neural network...")
            if self.selected_model_type == "rcnn":
                self.current_model = load_rcnn_model(self.selected_model, num_classes=7)
            else:
                self.current_model = YOLO(self.selected_model)
            self.current_model_path = self.selected_model
            self.current_model_type = self.selected_model_type

        # Open capture using seam factory
        if is_live:
            kw = getattr(self, "live_stream_kwargs", {})
            cap = open_capture(kw.get("mode", "listen"), **kw)
        else:
            cap = open_capture("file", self.selected_video)

        self.cap = cap
        is_live_cap = getattr(cap, "is_live", False)

        if not cap.isOpened():
            self._safe_log(active_listbox, "[ERROR] Unable to open video capture source.")
            self.stop_playback()
            return

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        scaled_focal_length = BASE_FOCAL_LENGTH_640 * (max(frame_w, 640) / 640.0)

        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0

        # Reporting Session Setup
        preset_info = PRESETS.get(getattr(self, "active_preset_name", ""), {})
        report_mod = preset_info.get("report_module")
        reporting_active = (report_mod is not None)
        report_token = 0

        conf_thresh_init = float(getattr(self, "active_conf_thresh", 0.25))

        if reporting_active:
            structure_id = None
            if report_mod == "bridge":
                structure_id = "Live-Bridge-Feed" if is_live else os.path.splitext(os.path.basename(self.selected_video))[0]
            
            source_type = "file" if not is_live_cap else getattr(cap, "mode", "http_listen")
            video_label = cap.display_name() if is_live_cap and hasattr(cap, "display_name") else self.selected_video

            report_token = self.reporter.start_session(
                video_label, self.current_model_path, self.current_model_type,
                fps, total_frames, conf_thresh_init,
                module=report_mod, structure_id=structure_id,
                is_live=is_live_cap, source_type=source_type
            )
            self._report_token = report_token

        frame_idx = 0
        self.fps_timer = time.time()
        self.fps_count = 0

        while self.is_running and cap.isOpened():
            if self.is_paused:
                time.sleep(0.05)
                continue

            frame_start = time.time()

            if not is_live_cap:
                speed_val = getattr(self, "playback_speed", 1.0)
                if speed_val > 1.2:
                    skip_count = int(round(speed_val)) - 1
                    for _ in range(skip_count):
                        if not cap.grab():
                            break
                        frame_idx += 1

            ret, frame = cap.read()
            if not ret:
                if is_live_cap:
                    state_str = getattr(cap, "state", "RECONNECTING")
                    self.root.after(0, self._update_live_stream_ui_state, state_str, cap.stats())
                    time.sleep(0.1)
                    continue
                else:
                    # Video demo playback complete
                    if reporting_active:
                        self.reporter.end_session(report_token)
                        self.reporter.generate_digest()
                    self._safe_log(active_listbox, "[INFO] Video playback complete.")
                    break

            if is_live_cap and (frame_w == 0 or frame_h == 0):
                frame_h, frame_w = frame.shape[:2]
                scaled_focal_length = BASE_FOCAL_LENGTH_640 * (max(frame_w, 640) / 640.0)

            raw_frame = frame.copy() if (reporting_active and self.current_model_type == "rcnn") else None
            report_dets = []

            frame_idx += 1
            conf_thresh = float(getattr(self, "active_conf_thresh", 0.25))

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

                            cx = (x1 + x2) / 2.0
                            norm_cx = cx / float(frame_w)
                            norm_w = width_px / float(frame_w)
                            norm_y2 = y2 / float(frame_h)

                            is_central_corridor = (0.32 <= norm_cx <= 0.68)
                            is_tunnel_or_wall = (norm_w > 0.40 and (x1 <= 50 or x2 >= frame_w - 50)) or not is_central_corridor

                            if is_tunnel_or_wall or not is_central_corridor:
                                has_lateral_feature = True
                                box_color = (0, 215, 255)  # Amber / Yellow
                                label_str = f"{class_name} {conf*100:.0f}% | {dist_m:.1f}m (Clearance)"
                                detected_items.append(f"{class_name} ({dist_m:.1f}m, Side/Tunnel)")
                                if dist_m < closest_lateral_dist:
                                    closest_lateral_dist = dist_m
                            else:
                                if dist_m < (WARNING_DISTANCE_CM / 100.0):
                                    box_color = (0, 0, 255)  # Red alert
                                    frame_alert = True
                                    label_str = f"{class_name} {conf*100:.0f}% | {dist_m:.1f}m [IN-TRACK ALERT]"
                                elif dist_m < 35.0:
                                    box_color = (0, 165, 255)  # Orange/Amber
                                    label_str = f"{class_name} {conf*100:.0f}% | {dist_m:.1f}m [APPROACHING]"
                                else:
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

            live_stats = cap.stats() if is_live_cap and hasattr(cap, "stats") else {}

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
                detected_items,
                is_live_cap,
                live_stats
            )

            if reporting_active:
                self.reporter.ingest(report_token, frame_idx, raw_frame if report_dets else None, report_dets)

            if not is_live_cap:
                speed_val = getattr(self, "playback_speed", 1.0)
                target_frame_time = (1.0 / (fps * speed_val)) if fps > 0 else 0.03
                loop_elapsed = time.time() - frame_start
                sleep_duration = max(0.001, target_frame_time - loop_elapsed)
                time.sleep(sleep_duration)

        if reporting_active:
            self.reporter.end_session(report_token)
            self.reporter.generate_digest()

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        self.root.after(0, self.stop_playback)

    def _update_live_stream_ui_state(self, state: str, stats: dict):
        if state == "LIVE":
            fps_str = f"{stats.get('measured_fps', 0.0):.1f}"
            self.live_state_badge.config(text=f"● LIVE ({fps_str} FPS)", bg="#134e4a", fg="#48bb78")
        elif state == "CONNECTING":
            self.live_state_badge.config(text="CONNECTING...", bg="#744210", fg="#fbd38d")
        elif state == "RECONNECTING":
            rc = stats.get("reconnects", 1)
            self.live_state_badge.config(text=f"⚠️ RECONNECTING ({rc})", bg="#742a2a", fg="#feb2b2")
            self.alert_label_live.config(
                text="📡 Live stream reconnecting - waiting for incoming frames...",
                bg="#744210",
                fg="#fbd38d"
            )
        else:
            self.live_state_badge.config(text=state, bg="#414868", fg="#c0caf5")

        age = stats.get("last_frame_age_s")
        age_str = f"{age:.1f}s" if age is not None else "--"
        self.lbl_live_ingest_stats.config(
            text=f"Ingest: {stats.get('measured_fps', 0.0):.1f} FPS | Drop: {stats.get('dropped', 0)} | Age: {age_str}"
        )

    # =========================================================================
    # UI Frame Render & Telemetry Update
    # =========================================================================
    def _update_ui_frame(
        self, frame_bgr, frame_idx, total_frames, obj_count,
        in_track_dist, lateral_dist, alert, lateral_feature, items,
        is_live_cap, live_stats
    ):
        canvas = self.video_canvas_live if is_live_cap else self.video_canvas_demo
        alert_lbl = self.alert_label_live if is_live_cap else self.alert_label_demo
        lbl_fps = self.lbl_fps_live if is_live_cap else self.lbl_fps_demo
        lbl_frame = self.lbl_frame_live if is_live_cap else self.lbl_frame_demo
        lbl_objects = self.lbl_objects_live if is_live_cap else self.lbl_objects_demo
        lbl_mname = self.lbl_metric_name_live if is_live_cap else self.lbl_metric_name_demo
        lbl_mval = self.lbl_metric_val_live if is_live_cap else self.lbl_metric_val_demo
        log_listbox = self.log_listbox_live if is_live_cap else self.log_listbox_demo

        # Update Live State Badge if live
        if is_live_cap and live_stats:
            self._update_live_stream_ui_state(live_stats.get("state", "LIVE"), live_stats)

        # Update Telemetry Readouts
        lbl_fps.config(text=f"{self.current_fps} FPS")
        if is_live_cap or total_frames <= 0:
            lbl_frame.config(text=f"LIVE #{frame_idx}")
        else:
            lbl_frame.config(text=f"{frame_idx} / {total_frames}")
        lbl_objects.config(text=str(obj_count))

        if self.mode == "distance_tracking":
            if in_track_dist < float("inf"):
                lbl_mname.config(text="Corridor Obstacle:")
                lbl_mval.config(text=f"{in_track_dist:.1f} m", fg="#ff4d4d" if alert else "#ffb74d")
            elif lateral_dist < float("inf"):
                lbl_mname.config(text="Lateral Clearance:")
                lbl_mval.config(text=f"{lateral_dist:.1f} m (Safe)", fg="#00e5ff")
            else:
                lbl_mname.config(text="Corridor Clearance:")
                lbl_mval.config(text="Clear Rail (Safe)", fg=COLOR_SUCCESS)
        elif self.mode == "track_defects":
            if alert:
                lbl_mval.config(text="DEFECT DETECTED", fg="#ff4d4d")
            elif obj_count > 0:
                lbl_mval.config(text="Minor Wear / Joint", fg="#ffb74d")
            else:
                lbl_mval.config(text="Clear Rail", fg=COLOR_SUCCESS)
        else:
            if obj_count > 0:
                lbl_mval.config(text=f"{obj_count} Anomaly", fg="#ffb74d")
            else:
                lbl_mval.config(text="Sound Structure", fg=COLOR_SUCCESS)

        # Update Alert Banner
        if self.mode == "distance_tracking":
            if alert:
                alert_lbl.config(
                    text="⚠️ CRITICAL BRAKING ALERT: OBSTACLE DIRECTLY ON TRACK (< 15m)!",
                    bg="#742a2a", fg="#feb2b2"
                )
            elif lateral_feature:
                alert_lbl.config(
                    text="ℹ️ Tunnel Portal / Lateral Clearance Maintained (Safe Corridor)",
                    bg="#2d3748", fg="#ffd700"
                )
            elif obj_count > 0:
                alert_lbl.config(
                    text="ℹ️ Track Corridor Feature Detected - Safe Stopping Margin",
                    bg="#2a4365", fg="#90cdf4"
                )
            else:
                alert_lbl.config(
                    text="✅ Track Clear & Sound - Normal Train Operations",
                    bg="#1c2d27", fg=COLOR_SUCCESS
                )
        elif self.mode == "track_defects":
            if alert:
                alert_lbl.config(
                    text=f"⚠️ DEFECT DETECTED: {', '.join(items[:2])}",
                    bg="#742a2a", fg="#feb2b2"
                )
            elif obj_count > 0:
                alert_lbl.config(
                    text=f"ℹ️ Track Observation: {', '.join(items[:2])}",
                    bg="#2a4365", fg="#90cdf4"
                )
            else:
                alert_lbl.config(
                    text="✅ Rail Surface Nominal - No Faults Detected",
                    bg="#1c2d27", fg=COLOR_SUCCESS
                )
        else:
            if alert or obj_count > 0:
                alert_lbl.config(
                    text=f"⚠️ Bridge Anomaly: {', '.join(items[:2])}",
                    bg="#742a2a", fg="#feb2b2"
                )
            else:
                alert_lbl.config(
                    text="✅ Bridge Structure Sound - No Anomalies Detected",
                    bg="#1c2d27", fg=COLOR_SUCCESS
                )

        # Log entries (throttled)
        if items and frame_idx % 5 == 0:
            msg = f"F#{frame_idx}: " + ", ".join(items[:2])
            log_listbox.insert(0, msg)
            if log_listbox.size() > 35:
                log_listbox.delete(tk.END)

        # Canvas rendering with aspect ratio preservation
        canvas_w = canvas.winfo_width()
        canvas_h = canvas.winfo_height()

        if canvas_w > 80 and canvas_h > 80:
            h, w, _ = frame_bgr.shape
            scale = min(canvas_w / w, canvas_h / h)
            nw, nh = int(w * scale), int(h * scale)
            resized = cv2.resize(frame_bgr, (nw, nh))

            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            self.photo = ImageTk.PhotoImage(image=img)

            canvas.delete("all")
            canvas.create_image(
                canvas_w // 2, canvas_h // 2,
                image=self.photo,
                anchor=tk.CENTER
            )

    def _on_close(self):
        self.stop_playback()
        if hasattr(self, "reporter"):
            self.reporter.shutdown()
        self.root.destroy()


def main():
    print("[INFO] Initializing Railway Infrastructure Inspection Suite...", flush=True)
    root = ctk.CTk()
    app = RailwayInspectorApp(root)
    root.deiconify()
    root.lift()
    root.focus_force()
    root.attributes('-topmost', True)
    root.after(1000, lambda: root.attributes('-topmost', False))
    print("[INFO] Inspection Suite dashboard ready! Running main loop...", flush=True)
    root.mainloop()


if __name__ == "__main__":
    main()
