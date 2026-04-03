"""
config.py - Load all configuration from .env file.

Provides a single `cfg` object used across the entire project.
Falls back to sensible defaults when a key is missing.

Usage:
    from config import cfg

    print(cfg.YOUTUBE_URL)
    print(cfg.YOLO_CONFIDENCE)
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Load .env file manually (no extra deps — works without python-dotenv)
def _load_env(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        logger.warning(".env not found at %s — using defaults", env_path.resolve())
        return
    with open(env_path, encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    logger.info(".env loaded from %s", env_path.resolve())


_load_env()


# ── Typed getters ─────────────────────────────────────────────────────────────

def _str(key: str, default: str) -> str:
    return os.environ.get(key, default)

def _int(key: str, default: int) -> int:
    v = os.environ.get(key)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        logger.warning("Config %s=%r is not an int — using default %d", key, v, default)
        return default

def _float(key: str, default: float) -> float:
    v = os.environ.get(key)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        logger.warning("Config %s=%r is not a float — using default %f", key, v, default)
        return default

def _bool(key: str, default: bool) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "on")

def _list(key: str, default: list[str]) -> list[str]:
    v = os.environ.get(key)
    if v is None:
        return default
    return [x.strip() for x in v.split(",") if x.strip()]

def _rect(key: str, default: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Parse 'x,y,w,h' string into a 4-int tuple."""
    v = os.environ.get(key)
    if v is None:
        return default
    try:
        parts = [int(p.strip()) for p in v.split(",")]
        if len(parts) != 4:
            raise ValueError("expected 4 values")
        return tuple(parts)
    except ValueError as e:
        logger.warning("Config %s=%r invalid rect (%s) — using default", key, v, e)
        return default


# ── Config object ─────────────────────────────────────────────────────────────

class Config:
    """
    All configuration values for the crossroads monitor.
    Edit .env to change any value — no Python edits needed.
    """

    # ── Flask / server ────────────────────────────────────────────────────
    HOST:          str   = _str("HOST",           "0.0.0.0")
    PORT:          int   = _int("PORT",            5000)
    DEBUG:         bool  = _bool("DEBUG",          False)

    # ── YouTube / camera ──────────────────────────────────────────────────
    YOUTUBE_URL:       str   = _str("YOUTUBE_URL",   "https://www.youtube.com/watch?v=1H0iTzv2jiQ")
    CAM_NAME:          str   = _str("CAM_NAME",      "intersection")
    CAM_FPS_LIMIT:     float = _float("CAM_FPS_LIMIT",    15.0)
    CAM_RECONNECT_DELAY: float = _float("CAM_RECONNECT_DELAY", 5.0)

    # Region rects: (x, y, width, height) in pixels
    RECT_NORTH: tuple = _rect("RECT_NORTH", (0,    0,    960, 540))
    RECT_EAST:  tuple = _rect("RECT_EAST",  (960,  0,    960, 540))
    RECT_SOUTH: tuple = _rect("RECT_SOUTH", (0,    540,  960, 540))
    RECT_WEST:  tuple = _rect("RECT_WEST",  (960,  540,  960, 540))

    # ── YOLO ─────────────────────────────────────────────────────────────
    YOLO_MODEL:         str   = _str("YOLO_MODEL",        "yolov5s")
    YOLO_CONFIDENCE:    float = _float("YOLO_CONFIDENCE",  0.45)
    YOLO_IOU:           float = _float("YOLO_IOU",         0.45)
    YOLO_FRAME_SKIP:    int   = _int("YOLO_FRAME_SKIP",    3)

    # Traffic light detector
    TLD_CONFIDENCE:     float = _float("TLD_CONFIDENCE",   0.35)
    TLD_FRAME_SKIP:     int   = _int("TLD_FRAME_SKIP",     3)
    TLD_MIN_PIXELS:     int   = _int("TLD_MIN_PIXELS",     30)

    # ── Image enhancer ────────────────────────────────────────────────────
    DENOISE_METHOD:     str   = _str("DENOISE_METHOD",     "bilateral")  # median|bilateral|wiener|none
    CLAHE_ENABLED:      bool  = _bool("CLAHE_ENABLED",     True)
    CLAHE_CLIP:         float = _float("CLAHE_CLIP",        2.5)
    CLAHE_AUTO:         bool  = _bool("CLAHE_AUTO",         True)
    BILATERAL_D:        int   = _int("BILATERAL_D",         7)
    BILATERAL_SIGMA:    float = _float("BILATERAL_SIGMA",   50.0)
    MEDIAN_KSIZE:       int   = _int("MEDIAN_KSIZE",        3)

    # ── DeepSORT ─────────────────────────────────────────────────────────
    DEEPSORT_MAX_AGE:   int   = _int("DEEPSORT_MAX_AGE",    30)
    DEEPSORT_N_INIT:    int   = _int("DEEPSORT_N_INIT",     3)
    DEEPSORT_MAX_DIST:  float = _float("DEEPSORT_MAX_DIST", 0.4)
    DEEPSORT_EMBEDDER:  str   = _str("DEEPSORT_EMBEDDER",   "mobilenet")

    # ── Car classifier (VMMR) ─────────────────────────────────────────────
    CLASSIFIER_CHECKPOINT:   str   = _str("CLASSIFIER_CHECKPOINT",   "clip_cars_checkpoint.pth")
    CLASSIFIER_CONFIDENCE:   float = _float("CLASSIFIER_CONFIDENCE",  0.30)
    CLASSIFIER_MIN_CROP:     int   = _int("CLASSIFIER_MIN_CROP",      48)
    CLASSIFIER_TTA_AUGMENTS: int   = _int("CLASSIFIER_TTA_AUGMENTS",  4)   # 1=off, 4=on
    CLASSIFIER_TTA_MIN_FRAMES: int = _int("CLASSIFIER_TTA_MIN_FRAMES", 10) # use TTA after N frames

    # Model vote accumulator
    VOTE_WINDOW:         int  = _int("VOTE_WINDOW",    10)
    VOTE_MIN_WIN:        int  = _int("VOTE_MIN_WIN",   3)

    # ── ANPR ──────────────────────────────────────────────────────────────
    ANPR_ENABLED:        bool       = _bool("ANPR_ENABLED",        True)
    ANPR_LANGUAGES:      list[str]  = _list("ANPR_LANGUAGES",      ["en"])
    ANPR_CONFIDENCE:     float      = _float("ANPR_CONFIDENCE",    0.45)
    ANPR_GPU:            bool       = _bool("ANPR_GPU",            True)

    # ── Crossroad manager ─────────────────────────────────────────────────
    MIN_GREEN_SECONDS:   float = _float("MIN_GREEN_SECONDS",   8.0)
    MAX_GREEN_SECONDS:   float = _float("MAX_GREEN_SECONDS",   45.0)
    YELLOW_SECONDS:      float = _float("YELLOW_SECONDS",      3.0)
    SCORE_SWITCH_RATIO:  float = _float("SCORE_SWITCH_RATIO",  1.8)

    # ── Kalman filter ─────────────────────────────────────────────────────
    KALMAN_LOOKAHEAD:    float = _float("KALMAN_LOOKAHEAD",    8.0)
    KALMAN_PROCESS_NOISE: float = _float("KALMAN_PROCESS_NOISE", 0.5)
    KALMAN_MEAS_NOISE:   float = _float("KALMAN_MEAS_NOISE",   1.0)

    # ── Car tracker ───────────────────────────────────────────────────────
    TRACKER_IOU_THRESH:   float = _float("TRACKER_IOU_THRESH",  0.30)
    TRACKER_EMBED_THRESH: float = _float("TRACKER_EMBED_THRESH", 0.80)
    TRACKER_MAX_LOST:     int   = _int("TRACKER_MAX_LOST",       25)
    TRACKER_MIN_CONFIRM:  int   = _int("TRACKER_MIN_CONFIRM",    2)

    # ── Logging ───────────────────────────────────────────────────────────
    LOG_LEVEL: str = _str("LOG_LEVEL", "INFO")

    def dump(self) -> dict:
        """Return all config values as a dict (useful for /config endpoint)."""
        return {
            k: v for k, v in vars(self.__class__).items()
            if not k.startswith("_") and not callable(v)
        }


cfg = Config()


# Apply log level immediately
logging.getLogger().setLevel(getattr(logging, cfg.LOG_LEVEL.upper(), logging.INFO))