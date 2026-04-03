"""
image_enhancer.py - Image pre-processing pipeline for crossroads video frames.

Applies in order:
  1. Noise reduction  : Median / Bilateral / Wiener (selectable)
  2. CLAHE            : Adaptive contrast enhancement (auto-regulates strength)

All operations work on BGR numpy arrays (OpenCV format).

Usage:
    enhancer = ImageEnhancer(
        denoise="bilateral",   # "median" | "bilateral" | "wiener" | "none"
        clahe=True,
        clahe_clip=2.0,        # 1.0-4.0; auto-reduced if over-enhancing
    )
    frame_clean = enhancer.process(frame_bgr)
    crop_clean  = enhancer.enhance_crop(crop_bgr)  # stronger for small crops
"""

from __future__ import annotations

import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImageEnhancer:
    """
    Configurable noise-reduction + contrast enhancement pipeline.

    Parameters
    ----------
    denoise : str
        "median"    - fast, good for salt & pepper noise
        "bilateral" - edge-preserving, best for general video noise
        "wiener"    - frequency-domain filter (requires scipy), best for blur
        "none"      - skip denoising
    clahe : bool
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    clahe_clip : float
        Initial clip limit (1.0-4.0). Automatically reduced per-frame
        if the frame is already well-lit (prevents over-enhancement).
    clahe_tile : tuple[int,int]
        Grid size for CLAHE. Smaller = more local enhancement.
    median_ksize : int
        Kernel size for median filter (must be odd, e.g. 3, 5).
    bilateral_d : int
        Diameter of each pixel neighbourhood for bilateral filter.
    bilateral_sigma : float
        Sigma for bilateral filter colour + space.
    auto_clahe : bool
        Dynamically adjust CLAHE clip based on frame brightness variance.
    """

    def __init__(
        self,
        denoise:        str   = "bilateral",
        clahe:          bool  = True,
        clahe_clip:     float = 2.0,
        clahe_tile:     tuple = (8, 8),
        median_ksize:   int   = 3,
        bilateral_d:    int   = 7,
        bilateral_sigma:float = 50.0,
        auto_clahe:     bool  = True,
    ):
        self.denoise         = denoise.lower()
        self.do_clahe        = clahe
        self.clahe_clip      = clahe_clip
        self.clahe_tile      = clahe_tile
        self.median_ksize    = median_ksize
        self.bilateral_d     = bilateral_d
        self.bilateral_sigma = bilateral_sigma
        self.auto_clahe      = auto_clahe

        # CLAHE objects (one per clip level to avoid re-creation)
        self._clahe_cache: dict[float, cv2.CLAHE] = {}

        # Check scipy for Wiener
        self._has_scipy = False
        if self.denoise == "wiener":
            try:
                from scipy.signal import wiener as _w
                self._wiener = _w
                self._has_scipy = True
            except ImportError:
                logger.warning("scipy not installed — falling back to bilateral filter")
                self.denoise = "bilateral"

        logger.info("ImageEnhancer | denoise=%s  clahe=%s  auto=%s",
                    self.denoise, clahe, auto_clahe)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Full pipeline for a camera region frame.
        Returns a new array — never modifies the original.
        """
        if frame is None or frame.size == 0:
            return frame
        out = self._denoise(frame)
        if self.do_clahe:
            clip = self._auto_clip(out) if self.auto_clahe else self.clahe_clip
            out  = self._apply_clahe(out, clip)
        return out

    def enhance_crop(self, crop: np.ndarray) -> np.ndarray:
        """
        Stronger pipeline for small vehicle/plate crops.
        Uses a tighter bilateral filter and slightly higher CLAHE clip.
        """
        if crop is None or crop.size == 0:
            return crop
        # Upscale tiny crops before enhancing
        h, w = crop.shape[:2]
        if h < 64 or w < 64:
            scale = max(64 / h, 64 / w)
            crop  = cv2.resize(crop, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_CUBIC)

        # Bilateral with tighter params for crop detail
        out = cv2.bilateralFilter(crop, d=5,
                                  sigmaColor=self.bilateral_sigma * 0.8,
                                  sigmaSpace=self.bilateral_sigma * 0.8)
        if self.do_clahe:
            clip = min(3.5, self._auto_clip(out) * 1.3)
            out  = self._apply_clahe(out, clip)
        return out

    # ------------------------------------------------------------------ #
    #  Denoise                                                             #
    # ------------------------------------------------------------------ #

    def _denoise(self, frame: np.ndarray) -> np.ndarray:
        if self.denoise == "median":
            return cv2.medianBlur(frame, self.median_ksize)

        if self.denoise == "bilateral":
            return cv2.bilateralFilter(
                frame,
                d=self.bilateral_d,
                sigmaColor=self.bilateral_sigma,
                sigmaSpace=self.bilateral_sigma,
            )

        if self.denoise == "wiener" and self._has_scipy:
            # Apply per-channel Wiener filter
            out = np.empty_like(frame)
            for c in range(frame.shape[2]):
                ch = frame[:, :, c].astype(np.float32)
                out[:, :, c] = np.clip(
                    self._wiener(ch, (5, 5)), 0, 255
                ).astype(np.uint8)
            return out

        return frame.copy()

    # ------------------------------------------------------------------ #
    #  CLAHE                                                               #
    # ------------------------------------------------------------------ #

    def _auto_clip(self, frame: np.ndarray) -> float:
        """
        Compute adaptive CLAHE clip limit based on frame luminance variance.

        Well-lit, high-contrast frames → lower clip (avoid over-enhancement).
        Dark, low-contrast frames      → higher clip (boost more aggressively).

        Returns a clip value in [1.0, clahe_clip].
        """
        gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_lum = float(np.mean(gray))
        std_lum  = float(np.std(gray))

        # Normalise std to [0,1]: higher std = more contrast already present
        norm_std = min(1.0, std_lum / 80.0)

        # Scale clip: dark/flat frames get full clip, bright/contrasty get 1.0
        clip = 1.0 + (self.clahe_clip - 1.0) * (1.0 - norm_std)

        # Extra reduction if very bright (avoid blowout)
        if mean_lum > 180:
            clip = max(1.0, clip * 0.7)

        return round(clip, 2)

    def _get_clahe(self, clip: float) -> cv2.CLAHE:
        key = round(clip, 1)
        if key not in self._clahe_cache:
            self._clahe_cache[key] = cv2.createCLAHE(
                clipLimit=key,
                tileGridSize=self.clahe_tile,
            )
        return self._clahe_cache[key]

    def _apply_clahe(self, frame: np.ndarray, clip: float) -> np.ndarray:
        """Apply CLAHE to L channel in LAB colour space (preserves hue)."""
        clahe = self._get_clahe(clip)
        lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq  = clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)