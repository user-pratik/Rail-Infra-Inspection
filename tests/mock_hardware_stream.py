"""
Mock hardware camera stream script.
Streams a video file in a continuous loop to an HTTP endpoint via ffmpeg,
simulating hardware drone / inspection camera feed.

Usage:
    python tests/mock_hardware_stream.py [video_path] [target_url]
"""

import os
import sys
import shutil
import subprocess
import time

def find_ffmpeg():
    bin_path = shutil.which("ffmpeg")
    if bin_path:
        return bin_path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    return "ffmpeg"

def start_mock_stream(
    video_path="sample_bridge_video.mp4",
    target_url="http://127.0.0.1:8554/live",
    target_fps=15
):
    if not os.path.isabs(video_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Check root or ModelWrappers
        candidates = [
            os.path.join(base_dir, video_path),
            os.path.join(base_dir, "ModelWrappers", "BridgeDefects", video_path),
            os.path.join(base_dir, "ModelWrappers", "TrackDefects", video_path)
        ]
        for c in candidates:
            if os.path.isfile(c):
                video_path = c
                break

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    ffmpeg_bin = find_ffmpeg()
    cmd = [
        ffmpeg_bin,
        "-re",
        "-stream_loop", "-1",
        "-i", video_path,
        "-f", "mpegts",
        "-vcodec", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        target_url
    ]

    print(f"[MOCK STREAM] Starting ffmpeg streaming: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    return proc

if __name__ == "__main__":
    vid = sys.argv[1] if len(sys.argv) > 1 else "sample_bridge_video.mp4"
    url = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8554/live"
    p = start_mock_stream(vid, url)
    try:
        p.wait()
    except KeyboardInterrupt:
        p.terminate()
        p.wait()
