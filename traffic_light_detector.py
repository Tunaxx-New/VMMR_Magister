"""
traffic_light_detector.py - Detects traffic lights in a frame and reads their color.

Strategy:
    1. YOLO detects the traffic light bounding box (COCO class 9)
    2. The crop is analyzed with HSV color masking to determine active color:
       red / yellow / green / unknown

Usage:
    from traffic_light_detector import TrafficLightDetector, LightColor

    tld = TrafficLightDetector()

    result = tld.detect(frame)
    if result:
        print(result.color)   # "red" | "yellow" | "green" | "unknown"
        print(result.confidence)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# COCO class ID for traffic light
TRAFFIC_LIGHT_CLASS_ID = 9


class LightColor(str, Enum):
    RED     = "red"
    YELLOW  = "yellow"
    GREEN   = "green"
    UNKNOWN = "unknown"


@dataclass
class TrafficLightDetection:
    color:       LightColor
    confidence:  float               # YOLO detection confidence
    color_score: float               # HSV color analysis score (0-1)
    bbox:        tuple[int, int, int, int]   # x1, y1, x2, y2 in original frame
    crop:        np.ndarray | None = field(default=None, repr=False)


class TrafficLightDetector:
    """
    Detects traffic lights with YOLO and reads color via HSV masking.

    Parameters
    ----------
    model_size : str
        YOLOv5 variant — same model used for cars, shared is fine.
    confidence : float
        YOLO minimum confidence for traffic light detections.
    frame_skip : int
        Run inference every N frames per direction.
    min_pixels : int
        Minimum number of colored pixels to trust a color read.
    """

    # HSV ranges for each light color
    # H is 0-179 in OpenCV, S and V are 0-255
    _HSV_RANGES = {
        LightColor.RED: [
            ((0,   120, 120), (10,  255, 255)),   # lower red
            ((165, 120, 120), (179, 255, 255)),   # upper red (wraps)
        ],
        LightColor.YELLOW: [
            ((15, 120, 120), (40, 255, 255)),
        ],
        LightColor.GREEN: [
            ((45, 80, 80), (90, 255, 255)),
        ],
    }

    def __init__(
        self,
        model_size: str = "yolov5s",
        confidence: float = 0.35,
        frame_skip: int = 3,
        min_pixels: int = 30,
    ):
        self.confidence = confidence
        self.frame_skip = frame_skip
        self.min_pixels = min_pixels
        self._frame_counters: dict[str, int] = {}

        from ultralytics import YOLO
        name_map = {
            "yolov5n": "yolov5nu.pt", "yolov5s": "yolov5su.pt",
            "yolov5m": "yolov5mu.pt", "yolov5l": "yolov5lu.pt",
            "yolov5x": "yolov5xu.pt",
        }
        pt_file = name_map.get(model_size, model_size)
        logger.info("TrafficLightDetector loading %s ...", pt_file)

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(pt_file)
        self._infer_kwargs = dict(
            device=device,
            conf=confidence,
            classes=[TRAFFIC_LIGHT_CLASS_ID],
            verbose=False,
        )
        logger.info("TrafficLightDetector ready | model=%s", pt_file)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def detect(
        self,
        frame: np.ndarray,
        direction: str = "unknown",
    ) -> list[TrafficLightDetection]:
        """
        Detect all traffic lights in a frame and return their colors.

        Returns a list — there may be 0, 1, or multiple traffic lights visible.
        """
        count = self._frame_counters.get(direction, 0)
        self._frame_counters[direction] = count + 1
        if count % self.frame_skip != 0:
            return []

        results = self.model(frame, imgsz=640, **self._infer_kwargs)
        detections: list[TrafficLightDetection] = []

        h, w = frame.shape[:2]
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                if int(box.cls[0]) != TRAFFIC_LIGHT_CLASS_ID:
                    continue

                conf = float(box.conf[0])
                x1 = max(0, int(box.xyxy[0][0]))
                y1 = max(0, int(box.xyxy[0][1]))
                x2 = min(w, int(box.xyxy[0][2]))
                y2 = min(h, int(box.xyxy[0][3]))

                crop = frame[y1:y2, x1:x2].copy() if (x2 > x1 and y2 > y1) else None
                color, color_score = self._read_color(crop)

                detections.append(TrafficLightDetection(
                    color=color,
                    confidence=conf,
                    color_score=color_score,
                    bbox=(x1, y1, x2, y2),
                    crop=crop,
                ))

        return detections

    def dominant_color(
        self,
        frame: np.ndarray,
        direction: str = "unknown",
    ) -> LightColor | None:
        """
        Convenience method — returns the color of the most confident
        traffic light in the frame, or None if none detected.
        """
        detections = self.detect(frame, direction)
        if not detections:
            return None
        best = max(detections, key=lambda d: d.confidence * d.color_score)
        return best.color if best.color != LightColor.UNKNOWN else None

    def draw(self, frame: np.ndarray, detections: list[TrafficLightDetection]) -> None:
        """Draw traffic light boxes on frame in-place."""
        color_bgr = {
            LightColor.RED:     (0,   0,   220),
            LightColor.YELLOW:  (0,   200, 220),
            LightColor.GREEN:   (0,   200, 60),
            LightColor.UNKNOWN: (160, 160, 160),
        }
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            bgr   = color_bgr[det.color]
            label = f"TL:{det.color.value} {det.confidence:.0%}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), bgr, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    #  Color analysis                                                      #
    # ------------------------------------------------------------------ #

    def _read_color(
        self,
        crop: np.ndarray | None,
    ) -> tuple[LightColor, float]:
        """
        Analyze a traffic light crop with HSV masking.
        Returns (color, score) where score is pixel_count / total_pixels.
        """
        if crop is None or crop.size == 0:
            return LightColor.UNKNOWN, 0.0

        # Focus on the top 2/3 of the light — the active lamp is usually
        # in the upper portion; the housing bottom adds noise
        h = crop.shape[0]
        roi = crop[:int(h * 0.85), :]

        hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        total = roi.shape[0] * roi.shape[1]
        if total == 0:
            return LightColor.UNKNOWN, 0.0

        scores: dict[LightColor, float] = {}

        for color, ranges in self._HSV_RANGES.items():
            mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            for (lo, hi) in ranges:
                mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))

            pixel_count = int(np.sum(mask > 0))
            if pixel_count >= self.min_pixels:
                scores[color] = pixel_count / total

        if not scores:
            return LightColor.UNKNOWN, 0.0

        best_color = max(scores, key=lambda c: scores[c])
        return best_color, scores[best_color]