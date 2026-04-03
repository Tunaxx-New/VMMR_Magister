"""
youtube_camera.py — YouTube live stream source split into 4 observable regions.

Install deps:
    pip install yt-dlp opencv-python numpy

Usage:
    cam = YouTubeCameraSource("https://www.youtube.com/watch?v=LIVE_ID", name="intersection")

    # Subscribe to the full frame
    cam.subscribe_full(my_full_frame_callback)

    # Subscribe to a specific region: 'top_left' | 'top_right' | 'bottom_left' | 'bottom_right'
    cam.subscribe_region('top_left',     north_callback)
    cam.subscribe_region('top_right',    east_callback)
    cam.subscribe_region('bottom_left',  south_callback)
    cam.subscribe_region('bottom_right', west_callback)

    cam.start()
    ...
    cam.stop()

Callbacks signature:
    def my_callback(name: str, region: str, frame: np.ndarray) -> None:
        ...
    def my_full_callback(name: str, frame: np.ndarray, regions: dict[str, np.ndarray]) -> None:
        ...
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Any

import cv2
import numpy as np
import yt_dlp

logger = logging.getLogger(__name__)

# Callback types
RegionCallback = Callable[[str, str, np.ndarray], None]        # (cam_name, region_name, frame)
FullCallback   = Callable[[str, np.ndarray, dict], None]        # (cam_name, full_frame, regions_dict)

REGIONS = ("top_left", "top_right", "bottom_left", "bottom_right")


# A rectangle defined as (x, y, width, height) in pixels.
# x/y are the top-left corner of the crop inside the full frame.
Rect = tuple[int, int, int, int]   # (x, y, w, h)


@dataclass
class RegionConfig:
    """
    Defines how to extract 4 named regions from a video frame.

    TWO MODES — pick one:

    ── Mode A: cross-split (default) ──────────────────────────────────────
    Splits the frame at a single (split_x, split_y) point into 4 quadrants.

        split_x=960, split_y=540  →  four 960×540 crops from a 1920×1080 frame

        +──────────────+──────────────+
        │  top_left    │  top_right   │
        │  (north)     │  (east)      │
        +──────────────+──────────────+
        │  bottom_left │  bottom_right│
        │  (south)     │  (west)      │
        +──────────────+──────────────+

    ── Mode B: custom rectangles ──────────────────────────────────────────
    Pass `rects` as a dict mapping region name → (x, y, width, height).
    Regions can overlap, be any size, and don't need to cover the full frame.

        rects={
            "top_left":     (0,   0,   960, 400),   # x, y, w, h
            "top_right":    (960, 0,   960, 400),
            "bottom_left":  (100, 400, 800, 300),
            "bottom_right": (1020,400, 800, 300),
        }

    Parameters
    ----------
    split_x : int | None
        Column pixel for the vertical dividing line (Mode A). None = centre.
    split_y : int | None
        Row pixel for the horizontal dividing line (Mode A). None = centre.
    padding : int
        Pixels to shave from every edge in Mode A (removes border noise).
    rects : dict[str, Rect] | None
        Custom (x, y, w, h) per region (Mode B). Overrides split_x/split_y.
    """
    split_x: int | None = None
    split_y: int | None = None
    padding: int = 0
    rects: dict[str, Rect] | None = None

    def slice(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        h, w = frame.shape[:2]

        # ── Mode B: custom rectangles ─────────────────────────────────────
        if self.rects:
            result = {}
            for name, (rx, ry, rw, rh) in self.rects.items():
                # clamp to frame bounds
                x1 = max(0, rx)
                y1 = max(0, ry)
                x2 = min(w, rx + rw)
                y2 = min(h, ry + rh)
                crop = frame[y1:y2, x1:x2]
                result[name] = crop
            return result

        # ── Mode A: cross-split ───────────────────────────────────────────
        cx = self.split_x if self.split_x is not None else w // 2
        cy = self.split_y if self.split_y is not None else h // 2
        p  = self.padding

        return {
            "top_left":     frame[p:cy-p,   p:cx-p],
            "top_right":    frame[p:cy-p,   cx+p:w-p],
            "bottom_left":  frame[cy+p:h-p, p:cx-p],
            "bottom_right": frame[cy+p:h-p, cx+p:w-p],
        }


class YouTubeCameraSource:
    """
    Pulls a YouTube live stream (or any yt-dlp-compatible URL),
    splits each frame into 4 regions, and notifies registered observers.

    Parameters
    ----------
    url : str
        YouTube watch URL, e.g. "https://www.youtube.com/watch?v=XXXX"
    name : str
        Identifier used in all callbacks.
    region_config : RegionConfig | None
        How to split the frame. Defaults to equal 4-quadrant split.
    fps_limit : float
        Max frames per second to emit to observers.
    reconnect_delay : float
        Seconds to wait before re-resolving the stream URL after failure.
    quality : str
        yt-dlp format string. Default picks best video ≤720p.
    """

    def __init__(
        self,
        url: str,
        name: str = "youtube",
        region_config: RegionConfig | None = None,
        fps_limit: float = 10.0,
        reconnect_delay: float = 5.0,
        quality: str = "best[height<=720][ext=mp4]/best[height<=720]/best",
    ):
        self.url = url
        self.name = name
        self.region_config = region_config or RegionConfig()
        self.fps_limit = fps_limit
        self.reconnect_delay = reconnect_delay
        self.quality = quality

        # { region_name -> [callbacks] }
        self._region_observers: dict[str, list[RegionCallback]] = {r: [] for r in REGIONS}
        self._full_observers:   list[FullCallback] = []
        self._lock = threading.Lock()

        self._cap: cv2.VideoCapture | None = None
        self._stream_url: str | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False

        self._frame_count = 0
        self._drop_count  = 0

    # ------------------------------------------------------------------ #
    #  Subscribe / unsubscribe                                             #
    # ------------------------------------------------------------------ #

    def subscribe_region(self, region: str, callback: RegionCallback) -> None:
        """
        Register a callback for one region.

        callback(cam_name: str, region: str, frame: np.ndarray) -> None
        """
        if region not in REGIONS:
            raise ValueError(f"region must be one of {REGIONS}, got {region!r}")
        with self._lock:
            if callback not in self._region_observers[region]:
                self._region_observers[region].append(callback)
        logger.debug("[%s] Subscribed to region '%s': %s", self.name, region, callback)

    def unsubscribe_region(self, region: str, callback: RegionCallback) -> None:
        with self._lock:
            self._region_observers[region] = [
                c for c in self._region_observers[region] if c != callback
            ]

    def subscribe_full(self, callback: FullCallback) -> None:
        """
        Register a callback for the full frame + all regions at once.

        callback(cam_name: str, full_frame: np.ndarray,
                 regions: dict[str, np.ndarray]) -> None
        """
        with self._lock:
            if callback not in self._full_observers:
                self._full_observers.append(callback)

    def unsubscribe_full(self, callback: FullCallback) -> None:
        with self._lock:
            self._full_observers = [c for c in self._full_observers if c != callback]

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def start(self) -> "YouTubeCameraSource":
        if self._running:
            logger.warning("[%s] Already running.", self.name)
            return self
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"yt-{self.name}",
            daemon=True,
        )
        self._thread.start()
        logger.info("[%s] Started → %s", self.name, self.url)
        return self

    def stop(self) -> None:
        self._stop_event.set()
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=8)
        self._release()
        logger.info("[%s] Stopped. frames=%d dropped=%d",
                    self.name, self._frame_count, self._drop_count)

    @property
    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------ #
    #  Flask MJPEG helpers                                                 #
    # ------------------------------------------------------------------ #

    def generate_mjpeg_full(self):
        """Stream the full frame as MJPEG."""
        yield from self._mjpeg_gen(lambda regions, full: full)

    def generate_mjpeg_region(self, region: str):
        """Stream a single region as MJPEG."""
        if region not in REGIONS:
            raise ValueError(f"Unknown region: {region!r}")
        yield from self._mjpeg_gen(lambda regions, full: regions[region])

    def _mjpeg_gen(self, pick):
        while not self._stop_event.is_set():
            frame = self._latest_full
            regions = self._latest_regions
            if frame is None:
                time.sleep(0.05)
                continue
            target = pick(regions, frame)
            if target is None or target.size == 0:
                time.sleep(0.05)
                continue
            _, buf = cv2.imencode(".jpg", target)
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + buf.tobytes()
                + b"\r\n"
            )
            time.sleep(1.0 / self.fps_limit)

    # ------------------------------------------------------------------ #
    #  Stats / debug                                                        #
    # ------------------------------------------------------------------ #

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "name":            self.name,
            "url":             self.url,
            "stream_url":      self._stream_url,
            "running":         self.is_running,
            "frames_emitted":  self._frame_count,
            "frames_dropped":  self._drop_count,
            "region_observers": {r: len(v) for r, v in self._region_observers.items()},
            "full_observers":   len(self._full_observers),
        }

    def debug_snapshot(self, path: str = "debug_snapshot.jpg") -> bool:
        """Save current full frame to disk for visual inspection."""
        if self._latest_full is None:
            return False
        frame = self._latest_full.copy()
        h, w = frame.shape[:2]
        cx = self.region_config.split_x or w // 2
        cy = self.region_config.split_y or h // 2
        cv2.line(frame, (cx, 0), (cx, h), (0, 255, 0), 2)
        cv2.line(frame, (0, cy), (w, cy), (0, 255, 0), 2)
        for name, (x, y) in [
            ("top_left",     (10, 30)),
            ("top_right",    (cx+10, 30)),
            ("bottom_left",  (10, cy+30)),
            ("bottom_right", (cx+10, cy+30)),
        ]:
            cv2.putText(frame, name, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imwrite(path, frame)
        logger.info("[%s] Snapshot saved to %s", self.name, path)
        return True

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    _latest_full:    np.ndarray | None = None
    _latest_regions: dict[str, np.ndarray] = {}

    def _resolve_stream_url(self) -> str | None:
        """
        Resolve a direct streamable URL from YouTube (VOD or LIVE).

        Live streams expose HLS (.m3u8) manifests — cv2.VideoCapture handles
        these natively via FFmpeg. We try clients in order of reliability for
        live content: android_vr → ios → android → web.
        """
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }

        # For live streams: accept HLS/DASH, don't filter by ext
        live_format  = "best[height<=720]/best"
        # For VOD: prefer direct mp4
        vod_format   = "best[height<=720][ext=mp4]/best[height<=720]/best"

        strategies: list[dict] = [
            # 1. android_vr — best for live streams, no bot checks
            {
                "quiet": True, "no_warnings": True,
                "format": live_format, "http_headers": headers,
                "extractor_args": {"youtube": {"player_client": ["android_vr"]}},
            },
            # 2. ios — reliable for live HLS
            {
                "quiet": True, "no_warnings": True,
                "format": live_format, "http_headers": headers,
                "extractor_args": {"youtube": {"player_client": ["ios"]}},
            },
            # 3. android — good fallback
            {
                "quiet": True, "no_warnings": True,
                "format": live_format, "http_headers": headers,
                "extractor_args": {"youtube": {"player_client": ["android"]}},
            },
            # 4. web + no skip filters — catches anything missed above
            {
                "quiet": True, "no_warnings": True,
                "format": live_format, "http_headers": headers,
                "extractor_args": {"youtube": {"player_client": ["web"]}},
            },
            # 5. bare — absolute fallback, let yt-dlp decide
            {"quiet": True, "no_warnings": True, "format": "best"},
        ]

        for i, opts in enumerate(strategies, 1):
            client = (opts.get("extractor_args") or {}).get("youtube", {}).get(
                "player_client", ["auto"])[0]
            try:
                logger.info("[%s] Strategy %d (%s) …", self.name, i, client)
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(self.url, download=False)

                # Top-level url (common for live HLS manifests)
                url = info.get("url")

                # If not at top level, search formats — prefer HLS for live
                if not url:
                    formats = info.get("formats") or []
                    is_live = info.get("is_live") or info.get("was_live")

                    if is_live:
                        # Prefer m3u8 manifest for live
                        for fmt in reversed(formats):
                            if (fmt.get("url") and
                                    fmt.get("vcodec") != "none" and
                                    "m3u8" in (fmt.get("url") or "")):
                                url = fmt["url"]
                                break

                    if not url:
                        # Fallback: any format with video
                        for fmt in reversed(formats):
                            if fmt.get("url") and fmt.get("vcodec") != "none":
                                url = fmt["url"]
                                break

                if url:
                    is_live = info.get("is_live") or info.get("was_live")
                    logger.info(
                        "[%s] Strategy %d succeeded — %s | live=%s | url=%.60s…",
                        self.name, i, client, bool(is_live), url,
                    )
                    return url

                logger.warning("[%s] Strategy %d (%s) returned no usable URL",
                               self.name, i, client)

            except Exception as exc:
                logger.warning("[%s] Strategy %d (%s) failed: %s",
                               self.name, i, client, exc)

        logger.error("[%s] All resolver strategies exhausted", self.name)
        return None

    def _open(self) -> bool:
        self._release()
        self._stream_url = self._resolve_stream_url()
        if not self._stream_url:
            return False
        self._cap = cv2.VideoCapture(self._stream_url)
        if not self._cap.isOpened():
            logger.error("[%s] cv2.VideoCapture failed to open stream", self.name)
            self._cap = None
            return False
        logger.info("[%s] Stream opened successfully", self.name)
        return True

    def _release(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None

    def _notify(self, full: np.ndarray, regions: dict[str, np.ndarray]) -> None:
        self._latest_full    = full
        self._latest_regions = regions

        with self._lock:
            full_obs   = list(self._full_observers)
            region_obs = {r: list(v) for r, v in self._region_observers.items()}

        for cb in full_obs:
            try:
                cb(self.name, full, regions)
            except Exception:
                logger.exception("[%s] Full observer error", self.name)

        for region, callbacks in region_obs.items():
            frame = regions.get(region)
            if frame is None or frame.size == 0:
                continue
            for cb in callbacks:
                try:
                    cb(self.name, region, frame)
                except Exception:
                    logger.exception("[%s][%s] Region observer error", self.name, region)

    def _capture_loop(self) -> None:
        min_interval = 1.0 / self.fps_limit

        while not self._stop_event.is_set():
            if not self._open():
                logger.warning("[%s] Retrying in %.1fs …", self.name, self.reconnect_delay)
                self._stop_event.wait(self.reconnect_delay)
                continue

            consecutive_fails = 0

            while not self._stop_event.is_set():
                t0 = time.perf_counter()

                ok, frame = self._cap.read()
                if not ok:
                    consecutive_fails += 1
                    self._drop_count  += 1
                    if consecutive_fails >= 10:
                        logger.warning("[%s] 10 consecutive read failures — reconnecting", self.name)
                        break
                    time.sleep(0.1)
                    continue

                consecutive_fails = 0
                self._frame_count += 1

                regions = self.region_config.slice(frame)
                self._notify(frame, regions)

                elapsed = time.perf_counter() - t0
                wait    = min_interval - elapsed
                if wait > 0:
                    time.sleep(wait)

            self._release()
            if not self._stop_event.is_set():
                logger.info("[%s] Reconnecting in %.1fs …", self.name, self.reconnect_delay)
                self._stop_event.wait(self.reconnect_delay)

    # ------------------------------------------------------------------ #
    #  Context manager                                                     #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "YouTubeCameraSource":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    def __repr__(self) -> str:
        return (f"YouTubeCameraSource(name={self.name!r}, "
                f"url={self.url!r}, running={self.is_running})")