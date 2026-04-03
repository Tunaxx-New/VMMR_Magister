"""
car_detector.py - YOLOv5 car detector for crossroads regions.

Install:
    pip install ultralytics opencv-python numpy

Usage:
    from car_detector import CarDetector

    detector = CarDetector(
        model_size="yolov5s",
        confidence=0.45,
        save_crops=True,
        crops_dir="detected_cars",
    )

    def on_north(cam_name, region, frame):
        detections = detector.detect(frame, direction="north")
        for det in detections:
            add_log("north", f"{det.class_name} detected ({det.confidence:.0%})", "info")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# COCO class IDs that count as vehicles
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class Detection:
    """Single detected vehicle in a frame."""
    class_id:   int
    class_name: str
    confidence: float
    bbox:       tuple[int, int, int, int]  # x1, y1, x2, y2
    direction:  str
    timestamp:  float = field(default_factory=time.time)
    crop:       np.ndarray | None = field(default=None, repr=False)

    @property
    def label(self) -> str:
        return f"{self.class_name} {self.confidence:.0%}"

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


class CarDetector:
    """
    Wraps ultralytics YOLOv5 for per-frame vehicle detection.

    Parameters
    ----------
    model_size : str
        YOLOv5 variant: 'yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'
        Or a direct path to a custom .pt file.
    confidence : float
        Minimum confidence threshold (0-1).
    iou_threshold : float
        NMS IoU threshold.
    device : str | None
        'cuda', 'cpu', or None for auto-detect.
    save_crops : bool
        Save a cropped image of each detected vehicle to disk.
    crops_dir : str
        Directory to save crop images (organised by direction).
    frame_skip : int
        Run inference every N frames per direction.
        1 = every frame, 3 = every 3rd frame (reduces CPU/GPU load).
    draw_boxes : bool
        Draw bounding boxes on the frame in-place so they appear in the stream.
    """

    # Maps short names to ultralytics YOLOv5 model filenames
    _MODEL_MAP = {
        "yolov5n": "yolov5nu.pt",
        "yolov5s": "yolov5su.pt",
        "yolov5m": "yolov5mu.pt",
        "yolov5l": "yolov5lu.pt",
        "yolov5x": "yolov5xu.pt",
    }

    def __init__(
        self,
        model_size: str = "yolov5s",
        confidence: float = 0.45,
        iou_threshold: float = 0.45,
        device: str | None = None,
        save_crops: bool = True,
        crops_dir: str = "detected_cars",
        frame_skip: int = 2,
        draw_boxes: bool = True,
    ):
        self.confidence    = confidence
        self.iou_threshold = iou_threshold
        self.save_crops    = save_crops
        self.crops_dir     = Path(crops_dir)
        self.frame_skip    = frame_skip
        self.draw_boxes    = draw_boxes
        self._frame_counters: dict[str, int] = {}
        self._last_detections: dict[str, list[Detection]] = {}
        self._last_detection_age: dict[str, int] = {}   # frames since last real detection
        self.cache_ttl_frames = frame_skip * 3           # expire after 3 inference cycles

        # Auto-detect device
        try:
            import torch
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            self.device = device or "cpu"

        # Resolve model filename
        pt_file = self._MODEL_MAP.get(model_size, model_size)

        logger.info("Loading YOLO model '%s' on %s ...", pt_file, self.device)

        from ultralytics import YOLO
        self.model = YOLO(pt_file)  # downloads automatically on first run

        # Store inference kwargs (passed on every predict call)
        self._infer_kwargs = dict(
            device=self.device,
            conf=confidence,
            iou=iou_threshold,
            classes=list(VEHICLE_CLASSES.keys()),
            verbose=False,
        )

        if self.save_crops:
            self.crops_dir.mkdir(parents=True, exist_ok=True)

        logger.info("YOLO ready | model=%s  device=%s  conf=%.2f",
                    pt_file, self.device, confidence)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def detect(
        self,
        frame: np.ndarray,
        direction: str = "unknown",
    ) -> list[Detection]:
        """Run YOLO on a BGR frame. Does NOT draw or touch the frame."""
        count = self._frame_counters.get(direction, 0)
        self._frame_counters[direction] = count + 1
        if count % self.frame_skip != 0:
            return []
        results = self.model(frame, imgsz=640, **self._infer_kwargs)
        return self._parse_results(results, frame, direction)

    def detect_and_annotate(
        self,
        frame: np.ndarray,
        direction: str = "unknown",
    ) -> tuple[list[Detection], np.ndarray]:
        """
        Return (new_detections, annotated_copy_of_frame).

        Every frame_skip-th frame: runs YOLO, updates cached boxes + age.
        All other frames: skips YOLO, redraws cached boxes on fresh copy.
        After cache_ttl_frames frames with no detection: boxes disappear.
        """
        annotated = frame.copy()

        count = self._frame_counters.get(direction, 0)
        self._frame_counters[direction] = count + 1
        run_inference = (count % self.frame_skip == 0)

        if run_inference:
            results    = self.model(frame, imgsz=640, **self._infer_kwargs)
            detections = self._parse_results(results, frame, direction)

            if detections:
                self._last_detections[direction]    = detections
                self._last_detection_age[direction] = 0
            else:
                age = self._last_detection_age.get(direction, 0) + 1
                self._last_detection_age[direction] = age
                if age > self.cache_ttl_frames:
                    self._last_detections.pop(direction, None)
        else:
            detections = []

        cached = self._last_detections.get(direction, [])
        if cached:
            self._draw(annotated, cached)

        return detections, annotated

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _parse_results(
        self,
        results,
        frame: np.ndarray,
        direction: str,
    ) -> list[Detection]:
        detections: list[Detection] = []
        h, w = frame.shape[:2]

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in VEHICLE_CLASSES:
                    continue

                conf = float(box.conf[0])
                x1   = max(0, int(box.xyxy[0][0]))
                y1   = max(0, int(box.xyxy[0][1]))
                x2   = min(w, int(box.xyxy[0][2]))
                y2   = min(h, int(box.xyxy[0][3]))

                crop = frame[y1:y2, x1:x2].copy() if (x2 > x1 and y2 > y1) else None

                det = Detection(
                    class_id=cls_id,
                    class_name=VEHICLE_CLASSES[cls_id],
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    direction=direction,
                    crop=crop,
                )
                detections.append(det)

                if self.save_crops and crop is not None:
                    self._save_crop(det)

        return detections

    def _save_crop(self, det: Detection) -> None:
        ts   = int(det.timestamp * 1000)
        name = f"{det.direction}_{det.class_name}_{ts}.jpg"
        path = self.crops_dir / det.direction
        path.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path / name), det.crop)

    def _draw(self, frame: np.ndarray, detections: list[Detection]) -> None:
        colors = {
            "car":        (0, 200, 255),
            "truck":      (0, 120, 255),
            "bus":        (0, 80,  200),
            "motorcycle": (180, 255, 0),
        }
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = colors.get(det.class_name, (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = det.label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                frame, label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 1, cv2.LINE_AA,
            )