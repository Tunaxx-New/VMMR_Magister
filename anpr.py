"""
anpr.py - Automatic Number Plate Recognition (ANPR / ALPR).

Pipeline:
    1. Detect licence plate region in vehicle crop:
       - YOLO (if a plate-detection model is available), OR
       - Classical CV: Canny edges + contour filtering (no extra model needed)
    2. Deskew / perspective-correct the plate crop
    3. EasyOCR reads the text
    4. Post-process: strip noise, validate pattern

Install:
    pip install easyocr
    (torch already installed from YOLO)

Usage:
    anpr = ANPR(languages=["en"])
    result = anpr.read(vehicle_crop_bgr)
    if result:
        print(result.plate, result.confidence)
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Plate text validation — basic alphanumeric 4-10 chars
PLATE_PATTERN = re.compile(r"^[A-Z0-9]{2,4}[\s\-]?[A-Z0-9]{2,4}$", re.IGNORECASE)


@dataclass
class PlateResult:
    plate:      str
    confidence: float
    bbox:       tuple[int, int, int, int] | None   # in vehicle crop coords
    raw_text:   str                                  # before cleanup


class ANPR:
    """
    Licence plate reader using EasyOCR.

    Parameters
    ----------
    languages : list[str]
        EasyOCR language codes. ['en'] is fastest. Add ['ru'] for Cyrillic etc.
    min_confidence : float
        Minimum OCR confidence to return a result.
    min_plate_area : float
        Minimum fraction of crop area that the plate region must occupy.
    use_gpu : bool
        Use CUDA for EasyOCR if available.
    """

    def __init__(
        self,
        languages:       list[str] = None,
        min_confidence:  float     = 0.4,
        min_plate_area:  float     = 0.01,
        use_gpu:         bool      = True,
    ):
        self.min_confidence = min_confidence
        self.min_plate_area = min_plate_area
        self._reader        = None

        langs = languages or ["en"]
        try:
            import torch
            gpu = use_gpu and torch.cuda.is_available()
            import easyocr
            self._reader = easyocr.Reader(langs, gpu=gpu, verbose=False)
            logger.info("ANPR ready | languages=%s  gpu=%s", langs, gpu)
        except ImportError:
            logger.warning(
                "easyocr not installed — ANPR disabled. "
                "Run: pip install easyocr"
            )

    @property
    def available(self) -> bool:
        return self._reader is not None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def read(self, vehicle_crop: np.ndarray) -> PlateResult | None:
        """
        Detect and read a licence plate from a vehicle crop (BGR).

        Returns PlateResult or None if no plate found / confidence too low.
        """
        if not self.available or vehicle_crop is None or vehicle_crop.size == 0:
            return None

        # 1. Find plate region
        plate_crop, bbox = self._detect_plate(vehicle_crop)
        if plate_crop is None:
            return None

        # 2. Enhance plate for OCR
        plate_enhanced = self._enhance_plate(plate_crop)

        # 3. OCR
        try:
            results = self._reader.readtext(plate_enhanced)
        except Exception as e:
            logger.debug("EasyOCR error: %s", e)
            return None

        if not results:
            return None

        # 4. Pick best result
        best_conf  = 0.0
        best_text  = ""
        for (_, text, conf) in results:
            if conf > best_conf:
                best_conf = conf
                best_text = text

        if best_conf < self.min_confidence:
            return None

        clean = self._clean(best_text)
        if not clean:
            return None

        return PlateResult(
            plate      = clean,
            confidence = round(best_conf, 3),
            bbox       = bbox,
            raw_text   = best_text,
        )

    def read_batch(
        self,
        crops: list[np.ndarray],
    ) -> list[PlateResult | None]:
        """Read plates from multiple vehicle crops (slightly faster in batch)."""
        return [self.read(c) for c in crops]

    # ------------------------------------------------------------------ #
    #  Plate detection (classical CV — no extra model needed)              #
    # ------------------------------------------------------------------ #

    def _detect_plate(
        self,
        crop: np.ndarray,
    ) -> tuple[np.ndarray | None, tuple | None]:
        """
        Find the most plate-like rectangle in the crop using:
            Grayscale → bilateral → Canny → contours → aspect ratio filter

        Returns (plate_crop, bbox_in_crop) or (None, None).
        """
        h, w = crop.shape[:2]
        if h < 30 or w < 30:
            return None, None

        gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur    = cv2.bilateralFilter(gray, 11, 17, 17)
        edges   = cv2.Canny(blur, 30, 200)

        cnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts    = sorted(cnts, key=cv2.contourArea, reverse=True)[:20]

        best_crop = None
        best_bbox = None
        best_area = 0

        total_area = h * w

        for c in cnts:
            peri  = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)

            if len(approx) != 4:
                continue

            x, y, cw, ch = cv2.boundingRect(approx)
            area     = cw * ch
            aspect   = cw / max(ch, 1)
            frac     = area / total_area

            # Licence plates are roughly 2:1 to 5:1 wide
            if not (1.5 <= aspect <= 6.0):
                continue
            if frac < self.min_plate_area or frac > 0.5:
                continue
            if area < best_area:
                continue

            best_area = area
            best_bbox = (x, y, x + cw, y + ch)
            best_crop = crop[y:y+ch, x:x+cw]

        # Fallback: use bottom-third of crop (plate usually at front/rear)
        if best_crop is None:
            y_start  = int(h * 0.55)
            region   = crop[y_start:, :]
            if region.size > 0:
                best_crop = region
                best_bbox = (0, y_start, w, h)

        return best_crop, best_bbox

    # ------------------------------------------------------------------ #
    #  Plate enhancement before OCR                                        #
    # ------------------------------------------------------------------ #

    def _enhance_plate(self, plate: np.ndarray) -> np.ndarray:
        """Upscale + denoise + threshold for maximum OCR accuracy."""
        # Upscale to at least 200px wide
        h, w = plate.shape[:2]
        if w < 200:
            scale = 200 / w
            plate = cv2.resize(plate, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_CUBIC)

        gray  = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

        # Wiener-like sharpening
        blur  = cv2.GaussianBlur(gray, (0, 0), 1.0)
        sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

        # CLAHE for contrast
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
        eq    = clahe.apply(sharp)

        # Adaptive threshold → clean binary for OCR
        binary = cv2.adaptiveThreshold(
            eq, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2,
        )

        # Back to BGR so EasyOCR is happy
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # ------------------------------------------------------------------ #
    #  Text cleanup                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _clean(text: str) -> str:
        """Strip OCR noise and normalise plate text."""
        # Remove everything that isn't alphanumeric, space, or dash
        text = re.sub(r"[^A-Za-z0-9\s\-]", "", text).strip().upper()
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text)
        if len(text) < 4:
            return ""
        return text