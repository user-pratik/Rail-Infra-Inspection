"""
Automated verification test for Phase 5 & Phase 6:
- Modern dark restyle with CustomTkinter
- Dual Tabview (Video Demo & Live Stream)
- Mutual exclusion between tabs
- Live stream wiring to HttpStreamCapture
- Presets 1-4 functional integrity
"""

import os
import sys
import time
import customtkinter as ctk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gui_app import RailwayInspectorApp, PRESETS

def test_gui_app_headless():
    print("\n==========================================")
    print("PHASE 5 & 6 VERIFICATION TEST")
    print("==========================================")

    root = ctk.CTk()
    root.withdraw()  # Headless mode for testing

    print("[STEP 1] Initializing RailwayInspectorApp with CustomTkinter...")
    app = RailwayInspectorApp(root)
    assert hasattr(app, "tabview"), "Missing tabview"
    assert "📹 Video Demo" in app.tabview._tab_dict, "Missing Video Demo tab"
    assert "📡 Live Stream" in app.tabview._tab_dict, "Missing Live Stream tab"
    print("Tabview and tabs verified.")

    print("\n[STEP 2] Verifying Mutual Exclusion between tabs...")
    # Initially both start buttons are normal
    assert str(app.btn_start_demo["state"]) == "normal", f"Expected normal, got {app.btn_start_demo['state']}"
    assert str(app.btn_start_live["state"]) == "normal", f"Expected normal, got {app.btn_start_live['state']}"

    # Simulate start on demo
    app.start_playback()
    assert app.is_running is True
    assert app.current_session_tab == "demo"
    assert str(app.btn_start_demo["state"]) == "disabled"
    assert str(app.btn_start_live["state"]) == "disabled"
    print("Verified: Starting Demo disables Live Stream Start button.")

    # Stop playback
    app.stop_playback()
    assert app.is_running is False
    assert str(app.btn_start_demo["state"]) == "normal"
    assert str(app.btn_start_live["state"]) == "normal"
    time.sleep(1.0)
    try:
        root.update()
    except Exception:
        pass
    app.cap = None
    print("Verified: Stopping Demo re-enables both Start buttons.")

    print("\n[STEP 3] Verifying Preset Switching on both tabs...")
    # Demo preset
    app.preset_var_demo.set("Railway Track Defect Inspection (Faster R-CNN)")
    app._on_preset_change_demo(None)
    assert app.mode == "track_defects"
    assert "Faster R-CNN" in app.selected_model or "RCNN" in app.selected_model

    # Live preset
    app.preset_var_live.set("Bridge Structural Defect Detection (YOLOv11)")
    app._on_preset_change_live(None)
    assert app.mode_live == "bridge_defects"
    assert "BridgeDefects" in app.selected_model_live
    print("Verified: Presets switch cleanly on both tabs.")

    print("\n[STEP 4] Verifying Live Stream Tab Integration with HttpStreamCapture...")
    app.live_mode_var.set("listen")
    app.live_port_entry.delete(0, "end")
    app.live_port_entry.insert(0, "8559")
    app.start_live_playback()
    assert app.is_running is True
    assert app.current_session_tab == "live"
    assert str(app.btn_start_demo["state"]) == "disabled"
    assert str(app.btn_start_live["state"]) == "disabled"

    # Wait briefly for background thread to load model and open capture
    t0 = time.time()
    while (app.cap is None or not getattr(app.cap, "is_live", False)) and time.time() - t0 < 8.0:
        try:
            root.update()
        except Exception:
            pass
        time.sleep(0.1)

    assert app.cap is not None, "app.cap was not initialized by worker thread"
    assert getattr(app.cap, "is_live", False) is True, f"Expected is_live=True, got {getattr(app.cap, 'is_live', None)}"
    print("Verified: Live stream capture initiated via open_capture factory.")

    app.stop_playback()
    assert app.is_running is False
    assert app.cap is None
    print("Verified: Live stream stopped and capture cleanly released.")

    root.destroy()
    print("\n=== PHASE 5 & 6 VERIFICATION PASSED! ===")
    return True

if __name__ == "__main__":
    ok = test_gui_app_headless()
    sys.exit(0 if ok else 1)
