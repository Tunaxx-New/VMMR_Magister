"""
car_tracker.py - Unique car identity tracking across frames and regions.

Each detected car gets a UUID on first sight and is re-identified by:
  1. IoU overlap (same region, consecutive frames)
  2. Cosine similarity of visual embeddings (cross-region / after occlusion)

The tracker is shared across all 4 regions.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from threading import Lock

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)

from vehicle_priority import get_priority
from config import cfg

# ── Tuning constants ──────────────────────────────────────────────────────────

# Thresholds loaded from cfg (.env)



EMBED_DIM = 512


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TrackedCar:
    id:             str
    model_name:     str                       # VMMR classification label
    direction:      str                       # current region direction
    bbox:           tuple[int, int, int, int] # last known bbox
    embedding:      np.ndarray                # L2-normalised 512-dim vector
    confidence:     float                     # classifier confidence
    first_seen:     float = field(default_factory=time.time)
    last_seen:      float = field(default_factory=time.time)
    frame_count:    int   = 0
    lost_frames:    int   = 0
    prev_direction: str   = ""
    is_new:         bool  = False             # True only on the announcing frame
    direction_changed: bool = False           # True only on the frame it moved

    priority_weight: float = 1.0    # from vehicle_priority.get_priority()
    priority_reason: str   = "standard vehicle"

    @property
    def wait_time(self) -> float:
        return self.last_seen - self.first_seen

    @property
    def short_id(self) -> str:
        return self.id[:6]


# ── Embedding extractor ───────────────────────────────────────────────────────

class EmbeddingExtractor:
    """
    Extracts a normalised 512-dim visual embedding from a BGR crop
    using the CLIP visual encoder inside CarClassifier.
    """

    _transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])

    def __init__(self, classifier):
        """
        classifier : CarClassifier instance (already loaded).
        Works with CLIP-based models; falls back to the full model minus head
        for other architectures.
        """
        self.device = classifier.device
        model       = classifier.model
        model_name  = classifier.model_name.lower()

        if model_name == "clip" and hasattr(model, "visual"):
            # Use CLIP visual encoder directly
            self._encoder = model.visual
            self._use_clip = True
        else:
            # For other architectures: strip the classification head,
            # use the penultimate layer as embedding
            self._encoder  = self._strip_head(model, model_name)
            self._use_clip = False

        self._encoder.eval()

    def _strip_head(self, model, name: str):
        """Return the backbone without the final classification layer."""
        import torch.nn as nn

        if name == "resnet" or name in ("simclrv2", "simclr", "simclr_v2"):
            model.fc = nn.Identity()
        elif name in ("efficientnet", "efficientnet_b0"):
            model.classifier = nn.Identity()
        elif name in ("mobilenet", "mobilenet_v3", "mobilenet_v3_large"):
            model.classifier = nn.Identity()
        elif name in ("vit", "vit_b_16", "vit_base"):
            model.heads = nn.Identity()
        elif name == "convnext":
            model.classifier = nn.Identity()
        elif name in ("alexnet", "vgg19"):
            model.classifier = nn.Identity()
        return model

    @torch.no_grad()
    def extract(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        """
        Returns an L2-normalised 1D numpy embedding, or None if crop is invalid.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        h, w = crop_bgr.shape[:2]
        if min(h, w) < 20:
            return None

        rgb    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil    = Image.fromarray(rgb)
        tensor = self._transform(pil).unsqueeze(0).to(self.device)

        if self._use_clip:
            # CLIP visual returns float16 on GPU — cast to float32
            feats = self._encoder(tensor.half() if next(
                self._encoder.parameters()).dtype == torch.float16 else tensor)
            feats = feats.float()
        else:
            feats = self._encoder(tensor)

        feats = feats.view(feats.size(0), -1)           # flatten
        feats = F.normalize(feats, dim=-1)               # L2 norm
        return feats.cpu().numpy()[0]

    @staticmethod
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two normalised vectors."""
        return float(np.dot(a, b))


# ── IoU helper ────────────────────────────────────────────────────────────────

def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


# ── Tracker ───────────────────────────────────────────────────────────────────

class CarTracker:
    """
    Maintains a registry of all unique cars seen across all regions.

    Thread-safe — update() is called from 4 different camera threads.
    """

    def __init__(self, extractor: EmbeddingExtractor):
        self.extractor = extractor
        self._cars: dict[str, TrackedCar] = {}   # id -> TrackedCar
        self._lock = Lock()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def update(
        self,
        direction: str,
        detections: list,              # list[Detection] from CarDetector
        crops:      list[np.ndarray],  # aligned with detections
    ) -> list[TrackedCar]:
        """
        Match raw detections to existing tracked cars or create new ones.

        Returns the list of TrackedCar objects active in this direction
        this frame (including those just confirmed as new).
        """
        with self._lock:
            # Step 0 — increment lost_frames for all cars in this direction
            for car in self._cars.values():
                if car.direction == direction:
                    car.lost_frames += 1

            active: list[TrackedCar] = []
            used_ids: set[str] = set()

            for det, crop in zip(detections, crops):
                embedding = self.extractor.extract(crop)
                bbox      = det.bbox
                matched   = self._match(direction, bbox, embedding, used_ids)

                if matched:
                    car = matched
                    used_ids.add(car.id)
                    car.bbox        = bbox
                    car.last_seen   = time.time()
                    car.lost_frames = 0
                    car.frame_count += 1
                    car.is_new      = False
                    car.direction_changed = False

                    if car.direction != direction:
                        car.prev_direction    = car.direction
                        car.direction         = direction
                        car.direction_changed = True

                    if embedding is not None:
                        # Exponential moving average of embedding for stability
                        car.embedding = 0.7 * car.embedding + 0.3 * embedding
                        car.embedding /= np.linalg.norm(car.embedding)
                else:
                    # New car — use placeholder until VMMR classifier returns a result
                    car_id = uuid.uuid4().hex[:8]
                    car = TrackedCar(
                        id          = car_id,
                        model_name  = "Identifying...",   # never show YOLO class name
                        direction   = direction,
                        bbox        = bbox,
                        embedding   = embedding if embedding is not None
                                      else np.zeros(EMBED_DIM),
                        confidence  = 0.0,   # 0 means "not yet classified"
                        frame_count = 1,
                        is_new      = False,  # wait for MIN_CONFIRM_FRAMES
                    )
                    self._cars[car_id] = car
                    used_ids.add(car_id)

                # Announce as new after MIN_CONFIRM_FRAMES
                if car.frame_count == cfg.TRACKER_MIN_CONFIRM:
                    car.is_new = True

                active.append(car)

            # Remove cars that have been lost too long
            to_remove = [
                cid for cid, c in self._cars.items()
                if c.lost_frames > cfg.TRACKER_MAX_LOST
            ]
            for cid in to_remove:
                logger.debug("Car %s left the scene", self._cars[cid].short_id)
                del self._cars[cid]

            return active

    # YOLO raw class names — never use these as the displayed model name
    _YOLO_CLASSES = {"car", "truck", "bus", "motorcycle", "vehicle"}

    def update_model_name(self, car_id: str, model_name: str, confidence: float) -> None:
        """
        Called when the VMMR classifier returns a result for a car.
        Only accepts real classifier results — ignores YOLO generic class names.
        """
        # Reject YOLO fallback labels
        if model_name.lower().strip() in self._YOLO_CLASSES:
            return
        # Reject empty or placeholder strings
        if not model_name or model_name == "Identifying...":
            return
        with self._lock:
            if car_id in self._cars:
                car = self._cars[car_id]
                if confidence > car.confidence:
                    car.model_name = model_name
                    car.confidence = confidence
                    weight, reason = get_priority(model_name)
                    car.priority_weight = weight
                    car.priority_reason = reason

    def all_cars(self) -> list[TrackedCar]:
        with self._lock:
            return list(self._cars.values())

    def cars_in(self, direction: str) -> list[TrackedCar]:
        with self._lock:
            return [c for c in self._cars.values()
                    if c.direction == direction and c.lost_frames == 0]

    def count(self, direction: str) -> int:
        return len(self.cars_in(direction))

    def longest_wait(self, direction: str) -> float:
        cars = self.cars_in(direction)
        return max((c.wait_time for c in cars), default=0.0)

    # ------------------------------------------------------------------ #
    #  Matching                                                            #
    # ------------------------------------------------------------------ #

    def _match(
        self,
        direction: str,
        bbox:      tuple,
        embedding: np.ndarray | None,
        used_ids:  set[str],
    ) -> TrackedCar | None:
        best_car   = None
        best_score = -1.0

        for car in self._cars.values():
            if car.id in used_ids:
                continue

            # Prefer IoU match in same direction (fast, cheap)
            if car.direction == direction:
                iou = _iou(car.bbox, bbox)
                if iou >= cfg.TRACKER_IOU_THRESH and iou > best_score:
                    best_score = iou
                    best_car   = car

            # Cross-region or re-entry: use embedding similarity
            elif (embedding is not None
                  and car.embedding is not None
                  and np.any(car.embedding)):
                sim = EmbeddingExtractor.cosine_sim(car.embedding, embedding)
                if sim >= cfg.TRACKER_EMBED_THRESH and sim > best_score:
                    best_score = sim
                    best_car   = car

        return best_car