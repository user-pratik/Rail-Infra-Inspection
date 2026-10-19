"""
Streaming capture module for Railway Infrastructure Inspection Suite.
Provides open_capture factory and HttpStreamCapture implementation.
"""

from __future__ import annotations

import collections
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.parse
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import yaml

log = logging.getLogger("streaming")

JPEG_SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"


class BaseStreamCapture:
    """Base class defining duck-typing interface for live captures matching cv2.VideoCapture."""
    is_live: bool = True
    state: str = "DISCONNECTED"

    def isOpened(self) -> bool:
        raise NotImplementedError

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        raise NotImplementedError

    def get(self, prop_id: int) -> float:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError

    def stats(self) -> Dict[str, Any]:
        raise NotImplementedError

    def display_name(self) -> str:
        raise NotImplementedError


class HttpStreamCapture(BaseStreamCapture):
    """
    Ultra-low latency HTTP stream capture using an FFmpeg subprocess pipeline.
    Supports both 'listen' (hardware POSTs to PC) and 'connect' (PC pulls from stream) modes.
    Decodes MJPEG pipe directly in memory with drop-oldest single-slot buffering.
    """

    def __init__(
        self,
        mode: str = "listen",
        host: str = "127.0.0.1",
        bind: str = "0.0.0.0",
        port: int = 8554,
        path: str = "/live",
        max_width: int = 1280,
        read_timeout: float = 5.0,
        source_url: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        self.is_live = True
        self.state = "DISCONNECTED"

        # Load yaml config if present
        if config_path is None:
            default_yaml = os.path.join(os.path.dirname(__file__), "stream.yaml")
            if os.path.isfile(default_yaml):
                config_path = default_yaml

        cfg: Dict[str, Any] = {}
        if config_path and os.path.isfile(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
            except Exception as e:
                log.warning(f"Failed to read stream config {config_path}: {e}")

        self.mode = mode or cfg.get("mode", "listen")
        self.host = host or cfg.get("host", "127.0.0.1")
        self.bind = bind or cfg.get("bind", "0.0.0.0")
        self.port = int(port or cfg.get("port", 8554))
        self.path = path or cfg.get("path", "/live")
        if not self.path.startswith("/"):
            self.path = "/" + self.path
        self.max_width = int(max_width or cfg.get("max_width", 1280))
        self.read_timeout = float(read_timeout or cfg.get("read_timeout", 5.0))

        if source_url:
            self.raw_url = source_url
            parsed = urllib.parse.urlsplit(source_url)
            if parsed.hostname:
                self.host = parsed.hostname
            if parsed.port:
                self.port = parsed.port
            if parsed.path:
                self.path = parsed.path
        else:
            if self.mode == "listen":
                self.raw_url = f"http://{self.bind}:{self.port}{self.path}"
            else:
                self.raw_url = f"http://{self.host}:{self.port}{self.path}"

        # Counters & Telemetry
        self.received_count = 0
        self.delivered_count = 0
        self.dropped_count = 0
        self.reconnect_count = 0
        self.measured_fps = 0.0
        self._last_fps_ts = 0.0
        self._fps_frame_count = 0
        self.last_frame_ts = 0.0
        self._width = 0.0
        self._height = 0.0

        # Diagnostics & Stderr
        self._stderr_lines: collections.deque[str] = collections.deque(maxlen=20)

        # Thread synchronization
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._new_frame: Optional[np.ndarray] = None
        self._has_new_frame = False
        self._stopped = False
        self._process: Optional[subprocess.Popen] = None
        self._use_fallback_cv2 = False
        self._cv2_cap: Optional[cv2.VideoCapture] = None

        # Check ffmpeg availability
        self._ffmpeg_bin = shutil.which("ffmpeg")
        if not self._ffmpeg_bin:
            # Check imageio_ffmpeg
            try:
                import imageio_ffmpeg
                self._ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                self._ffmpeg_bin = None

        if not self._ffmpeg_bin:
            if self.mode == "listen":
                raise RuntimeError(
                    "Listen mode requires ffmpeg on PATH. Please install ffmpeg via "
                    "'winget install Gyan.FFmpeg' or 'pip install imageio-ffmpeg'."
                )
            else:
                log.warning("FFmpeg binary not found. Falling back to OpenCV CAP_FFMPEG backend.")
                self._use_fallback_cv2 = True

        self._worker_thread = threading.Thread(target=self._run_loop, name="stream-reader", daemon=True)
        self._worker_thread.start()

    def display_name(self) -> str:
        """Returns URL with credentials and query string masked."""
        try:
            parsed = urllib.parse.urlsplit(self.raw_url)
            netloc = parsed.netloc
            if "@" in netloc:
                auth, host_port = netloc.split("@", 1)
                netloc = f"***:***@{host_port}"
            # Mask or strip query
            query = "" if not parsed.query else "..."
            masked = urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, query, ""))
            return masked
        except Exception:
            return f"http://***:{self.port}{self.path}"

    def isOpened(self) -> bool:
        return not self._stopped

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Wait up to read_timeout for the newest unseen frame (drop-oldest).
        Returns (True, frame) or (False, None) on timeout/stop.
        """
        deadline = time.time() + self.read_timeout
        with self._cond:
            while not self._has_new_frame and not self._stopped:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return False, None
                self._cond.wait(timeout=min(0.2, remaining))

            if self._stopped:
                return False, None

            if self._has_new_frame and self._new_frame is not None:
                frame = self._new_frame
                self._has_new_frame = False
                self.delivered_count += 1
                return True, frame

            return False, None

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        elif prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return 0.0
        elif prop_id == cv2.CAP_PROP_FPS:
            return float(self.measured_fps) if self.measured_fps > 0 else 25.0
        elif prop_id == cv2.CAP_PROP_POS_FRAMES:
            return float(self.delivered_count)
        return 0.0

    def release(self) -> None:
        self._stopped = True
        with self._cond:
            self._cond.notify_all()
        self._terminate_process()
        if self._cv2_cap is not None:
            try:
                self._cv2_cap.release()
            except Exception:
                pass
            self._cv2_cap = None
        self.state = "DISCONNECTED"

    def stats(self) -> Dict[str, Any]:
        now = time.time()
        age = round(now - self.last_frame_ts, 2) if self.last_frame_ts > 0 else None
        return {
            "mode": self.mode,
            "url": self.display_name(),
            "state": self.state,
            "received": self.received_count,
            "delivered": self.delivered_count,
            "dropped": self.dropped_count,
            "reconnects": self.reconnect_count,
            "measured_fps": round(self.measured_fps, 1),
            "last_frame_age_s": age,
        }

    # ------------------------------------------------------------------ Internal Loop
    def _terminate_process(self) -> None:
        p = self._process
        self._process = None
        if p is not None:
            try:
                p.terminate()
                p.wait(timeout=1.0)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        try:
            if proc.stderr:
                for line in iter(proc.stderr.readline, b""):
                    if not line:
                        break
                    text = line.decode("utf-8", errors="replace").strip()
                    if text:
                        self._stderr_lines.append(text)
        except Exception:
            pass

    def _build_ffmpeg_cmd(self) -> list[str]:
        cmd = [
            self._ffmpeg_bin or "ffmpeg",
            "-hide_banner",
            "-loglevel", "warning",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-probesize", "32768",
            "-analyzeduration", "0",
        ]
        if self.mode == "listen":
            cmd.extend(["-listen", "1"])

        cmd.extend([
            "-i", self.raw_url,
            "-an",
            "-vf", f"scale='min({self.max_width},iw)':-2",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-q:v", "3",
            "pipe:1",
        ])
        return cmd

    def _run_loop(self) -> None:
        backoff_steps = [1.0, 2.0, 4.0, 8.0, 10.0]
        backoff_idx = 0

        while not self._stopped:
            if self._use_fallback_cv2:
                self._run_cv2_fallback()
            else:
                self._run_ffmpeg_pipeline()

            if self._stopped:
                break

            self.state = "RECONNECTING"
            self.reconnect_count += 1
            sleep_time = backoff_steps[min(backoff_idx, len(backoff_steps) - 1)]
            backoff_idx += 1
            log.info(f"Stream {self.display_name()} disconnected. Reconnecting in {sleep_time}s...")

            # Sleep with early abort if released
            stop_deadline = time.time() + sleep_time
            while time.time() < stop_deadline and not self._stopped:
                time.sleep(0.1)

    def _run_ffmpeg_pipeline(self) -> None:
        cmd = self._build_ffmpeg_cmd()
        self.state = "CONNECTING"
        log.info(f"Starting ffmpeg stream ({self.mode} mode): {self.display_name()}")

        proc: Optional[subprocess.Popen] = None
        try:
            # On Windows, suppress console window creation
            creationflags = 0
            if os.name == "nt":
                creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0x08000000

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**6,
                creationflags=creationflags,
            )
            self._process = proc

            stderr_thread = threading.Thread(target=self._drain_stderr, args=(proc,), daemon=True)
            stderr_thread.start()

            stdout = proc.stdout
            if not stdout:
                return

            raw_buffer = bytearray()
            chunk_size = 32768

            while not self._stopped and proc.poll() is None:
                chunk = stdout.read(chunk_size)
                if not chunk:
                    break

                raw_buffer.extend(chunk)

                # Extract all complete JPEG frames in buffer
                while True:
                    soi = raw_buffer.find(JPEG_SOI)
                    if soi == -1:
                        # No SOI found, retain only last byte if it might be 0xff
                        if len(raw_buffer) > 1:
                            raw_buffer = raw_buffer[-1:]
                        break

                    if soi > 0:
                        del raw_buffer[:soi]
                        soi = 0

                    eoi = raw_buffer.find(JPEG_EOI, 2)
                    if eoi == -1:
                        # Incomplete frame, wait for next chunk
                        break

                    # Complete JPEG frame found
                    jpeg_bytes = bytes(raw_buffer[: eoi + 2])
                    del raw_buffer[: eoi + 2]

                    # Decode JPEG
                    frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None and frame.size > 0:
                        self._on_frame_received(frame)

        except Exception as e:
            log.error(f"Error in stream pipeline: {e}")
        finally:
            self._terminate_process()

    def _run_cv2_fallback(self) -> None:
        self.state = "CONNECTING"
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "fflags;nobuffer|flags;low_delay"
        cap = cv2.VideoCapture(self.raw_url, cv2.CAP_FFMPEG)
        self._cv2_cap = cap

        if not cap.isOpened():
            return

        while not self._stopped and cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            self._on_frame_received(frame)

        cap.release()
        self._cv2_cap = None

    def _on_frame_received(self, frame: np.ndarray) -> None:
        now = time.time()
        self.state = "LIVE"
        self.last_frame_ts = now
        self.received_count += 1
        h, w = frame.shape[:2]
        self._width = float(w)
        self._height = float(h)

        # FPS calculation (EMA / rolling)
        if self._last_fps_ts == 0.0:
            self._last_fps_ts = now
            self._fps_frame_count = 1
        else:
            self._fps_frame_count += 1
            dt_fps = now - self._last_fps_ts
            if dt_fps >= 1.0:
                current = self._fps_frame_count / dt_fps
                if self.measured_fps <= 0:
                    self.measured_fps = current
                else:
                    self.measured_fps = 0.7 * self.measured_fps + 0.3 * current
                self._last_fps_ts = now
                self._fps_frame_count = 0

        # Drop-oldest single-slot buffering
        with self._cond:
            if self._has_new_frame:
                self.dropped_count += 1
            self._new_frame = frame
            self._has_new_frame = True
            self._cond.notify()


def open_capture(kind: str = "file", source: Any = None, **kwargs) -> Any:
    """
    Factory function for frame captures.
    - kind="file": returns standard cv2.VideoCapture(source), unchanged.
    - kind in ("listen", "connect", "http_listen", "http_connect", "live"): returns HttpStreamCapture.
    """
    if kind == "file":
        return cv2.VideoCapture(str(source) if source else "")
    elif kind in ("listen", "connect", "http_listen", "http_connect", "live"):
        mode = "listen" if kind in ("listen", "http_listen") else "connect"
        # If kwargs already specifies mode, honor it
        actual_mode = kwargs.pop("mode", mode)
        source_url = str(source) if source else kwargs.pop("source_url", None)
        return HttpStreamCapture(mode=actual_mode, source_url=source_url, **kwargs)
    else:
        raise ValueError(f"Unknown capture kind: {kind}")
