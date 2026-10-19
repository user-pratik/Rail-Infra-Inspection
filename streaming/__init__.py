"""
Streaming package for Railway Infrastructure Inspection System.
Provides capture factories and HttpStreamCapture for live stream ingestion.
"""

from .capture import open_capture, HttpStreamCapture, BaseStreamCapture

__all__ = ["open_capture", "HttpStreamCapture", "BaseStreamCapture"]
