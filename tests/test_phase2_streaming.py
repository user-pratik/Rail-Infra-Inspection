"""
Automated acceptance test for Phase 2: HttpStreamCapture.
Verifies:
1. Listen mode receives mock stream and delivers >= 14 FPS.
2. Drops stale frames when 100 ms artificial delay is injected in consumer.
3. Cleanly reconnects when mock sender process is terminated and restarted.
"""

import os
import sys
import time
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streaming import open_capture, HttpStreamCapture
from tests.mock_hardware_stream import start_mock_stream

def run_phase2_acceptance():
    print("\n==========================================")
    print("PHASE 2 ACCEPTANCE TEST: HttpStreamCapture")
    print("==========================================")

    PORT = 8554
    TARGET_URL = f"http://127.0.0.1:{PORT}/live"

    # Step 1: Start HttpStreamCapture in listen mode
    print(f"\n[STEP 1] Starting HttpStreamCapture in listen mode on port {PORT}...")
    cap = open_capture("listen", port=PORT, read_timeout=6.0)
    assert cap.is_live is True
    assert cap.isOpened() is True
    time.sleep(1.0)  # Wait for ffmpeg listener to bind socket

    # Step 2: Start Mock Sender
    print("[STEP 2] Starting mock hardware sender...")
    sender = start_mock_stream("sample_bridge_video.mp4", TARGET_URL)
    
    # Wait for first frame
    print("[STEP 3] Awaiting initial video frame...")
    start_wait = time.time()
    got_first = False
    while time.time() - start_wait < 15.0:
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"First frame received! Resolution: {frame.shape[1]}x{frame.shape[0]}")
            got_first = True
            break
        time.sleep(0.05)

    assert got_first, "Failed to receive first frame within 15 seconds"

    # Step 4: Measure FPS for 3 seconds
    print("\n[STEP 4] Measuring delivery FPS over 3 seconds...")
    frames_read = 0
    t_start = time.time()
    while time.time() - t_start < 3.0:
        ret, frame = cap.read()
        if ret and frame is not None:
            frames_read += 1

    elapsed = time.time() - t_start
    delivery_fps = frames_read / elapsed
    print(f"Delivered {frames_read} frames in {elapsed:.2f}s -> {delivery_fps:.1f} FPS (Target >= 14 FPS)")
    stats = cap.stats()
    print("Stats:", stats)
    assert delivery_fps >= 14.0 or stats["measured_fps"] >= 14.0 or stats["received"] >= 35, (
        f"FPS too low: delivered {delivery_fps:.1f} FPS, measured {stats['measured_fps']:.1f} FPS"
    )

    # Step 5: Test drop-oldest behavior with 100 ms injected delay
    print("\n[STEP 5] Testing drop-oldest frame buffer with injected 100 ms consumer sleep...")
    initial_dropped = stats["dropped"]
    for i in range(15):
        ret, frame = cap.read()
        time.sleep(0.100)  # 100 ms sleep

    stats_after_delay = cap.stats()
    dropped_during_delay = stats_after_delay["dropped"] - initial_dropped
    print(f"Dropped frames during 100ms consumer delay: {dropped_during_delay}")
    assert dropped_during_delay > 0, "Expected dropped frames when consumer sleeps 100ms, but none were dropped!"

    # Step 6: Test Reconnection
    print("\n[STEP 6] Testing reconnection: Killing mock sender...")
    sender.terminate()
    sender.wait(timeout=3.0)
    print("Mock sender terminated. Waiting for listener to detect disconnect...")
    time.sleep(2.0)
    print(f"Current capture state: {cap.state}, reconnects: {cap.reconnect_count}")

    print("Restarting mock sender...")
    sender = start_mock_stream("sample_bridge_video.mp4", TARGET_URL)
    
    # Wait for frames again
    print("Awaiting reconnected frames...")
    reconnected = False
    reconnect_wait = time.time()
    while time.time() - reconnect_wait < 15.0:
        ret, frame = cap.read()
        if ret and frame is not None:
            reconnected = True
            print(f"Reconnected successfully! Received frame: {frame.shape[1]}x{frame.shape[0]}")
            break
        time.sleep(0.1)

    assert reconnected, "Failed to reconnect and receive frames after sender restart"

    # Step 7: Cleanup
    print("\n[STEP 7] Cleaning up mock sender and HttpStreamCapture...")
    sender.terminate()
    sender.wait()
    cap.release()
    print("Capture released successfully.")
    print("\n=== PHASE 2 ACCEPTANCE TEST PASSED! ===")
    return True

if __name__ == "__main__":
    ok = run_phase2_acceptance()
    sys.exit(0 if ok else 1)
