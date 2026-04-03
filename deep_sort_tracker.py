"""
deep_sort_tracker.py - DeepSORT-based vehicle tracking wrapper.

Replaces the manual IoU + embedding tracker in car_tracker.py with
the battle-tested DeepSORT algorithm:
    - Kalman filter  : smooth bbox prediction between detections
    - Appearance re-ID: deep embedding matching across frames / occlusions
    - Hungarian algorithm: globally optimal detection-to-track assignment

Install:
    pip install deep-sort-realtime

Usage:
    tracker = DeepSortTracker()
    tracks  = tracker.update(detections, frame)
    for t in tracks:
        print(t.track_id, t.bbox, t.state)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Unified track object returned by DeepSortTracker.update()."""
    track_id:    int
    bbox:        tuple[int, int, int, int]   # x1, y1, x2, y2
    confidence:  float
    class_name:  str
    direction:   str
    age:         int                          # frames since first detection
    hits:        int                          # confirmed detection count
    is_new:      bool = False
    crop:        np.ndarray | None = field(default=None, repr=False)


class DeepSortTracker:
    """
    Wraps deep_sort_realtime.DeepSort for multi-object vehicle tracking.

    Parameters
    ----------
    max_age : int
        Frames a track survives without a matching detection.
    n_init : int
        Detections needed to confirm a new track.
    max_cosine_distance : float
        Re-ID embedding distance threshold (0 = identical, 1 = different).
    embedder : str
        Appearance feature extractor: 'mobilenet' | 'torchreid' | 'clip_ViT-B/16'
    """

    def __init__(
        self,
        max_age:             int   = 30,
        n_init:              int   = 3,
        max_cosine_distance: float = 0.4,
        embedder:            str   = "mobilenet",
    ):
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self._ds = DeepSort(
                max_age            = max_age,
                n_init             = n_init,
                max_cosine_distance= max_cosine_distance,
                embedder           = embedder,
                half               = False,
                embedder_gpu       = False,
            )
            self._available = True
            logger.info("DeepSORT initialised | max_age=%d n_init=%d embedder=%s",
                        max_age, n_init, embedder)
        except ImportError:
            logger.warning(
                "deep-sort-realtime not installed. "
                "Run: pip install deep-sort-realtime\n"
                "Falling back to IoU tracker."
            )
            self._available = False

        self._prev_ids:   set[int] = set()
        self._lock        = Lock()

    @property
    def available(self) -> bool:
        return self._available

    def update(
        self,
        detections: list,          # list[Detection] from CarDetector
        frame:      np.ndarray,
        direction:  str = "unknown",
    ) -> list[Track]:
        """
        Feed new detections into DeepSORT and return confirmed tracks.

        Parameters
        ----------
        detections : list[Detection]
            Raw YOLOv5 detections for this frame/region.
        frame : np.ndarray
            Full BGR frame (DeepSORT uses it for appearance embedding).
        direction : str
            Compass label for this region.

        Returns
        -------
        list[Track]
        """
        if not self._available or not detections:
            # Return simple tracks from raw detections
            return self._fallback_tracks(detections, direction)

        with self._lock:
            # Convert to DeepSORT format: [[x1,y1,w,h], conf, class_label]
            ds_dets = []
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                w = x2 - x1
                h = y2 - y1
                if w > 0 and h > 0:
                    ds_dets.append(([x1, y1, w, h], det.confidence, det.class_name))

            if not ds_dets:
                return []

            raw_tracks = self._ds.update_tracks(ds_dets, frame=frame)

        tracks: list[Track] = []
        current_ids: set[int] = set()

        for rt in raw_tracks:
            if not rt.is_confirmed():
                continue

            ltrb = rt.to_ltrb()
            x1, y1, x2, y2 = (
                max(0, int(ltrb[0])),
                max(0, int(ltrb[1])),
                int(ltrb[2]),
                int(ltrb[3]),
            )
            h_f, w_f = frame.shape[:2]
            x2 = min(w_f, x2)
            y2 = min(h_f, y2)
            crop = frame[y1:y2, x1:x2].copy() if (x2 > x1 and y2 > y1) else None

            tid = int(rt.track_id)
            current_ids.add(tid)

            track = Track(
                track_id   = tid,
                bbox       = (x1, y1, x2, y2),
                confidence = rt.det_conf or 0.0,
                class_name = rt.det_class or "car",
                direction  = direction,
                age        = rt.age,
                hits       = rt.hits,
                is_new     = (tid not in self._prev_ids),
                crop       = crop,
            )
            tracks.append(track)

        self._prev_ids = current_ids
        return tracks

    def _fallback_tracks(self, detections, direction: str) -> list[Track]:
        """Simple pass-through when DeepSORT is unavailable."""
        tracks = []
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox
            tracks.append(Track(
                track_id   = id(det) % 100000,
                bbox       = det.bbox,
                confidence = det.confidence,
                class_name = det.class_name,
                direction  = direction,
                age        = 1,
                hits       = 1,
                is_new     = True,
                crop       = det.crop,
            ))
        return tracks