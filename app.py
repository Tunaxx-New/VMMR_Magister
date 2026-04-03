"""
app.py - Crossroads monitor — full pipeline.

Frame pipeline per region:
    YouTube stream
        -> ImageEnhancer  (Bilateral/Median/Wiener + adaptive CLAHE)
        -> YOLOv5         (vehicle + traffic light detection)
        -> DeepSORT       (multi-object tracking with Kalman + re-ID)
        -> CarTracker     (unique IDs, cross-region re-ID, priority weights)
        -> CarClassifier  (VMMR model name)
        -> ANPR           (licence plate OCR)
        -> CrossroadManager (Kalman-filtered priority scores, auto phase switching)
        -> Dashboard      (MJPEG stream + SocketIO events)
"""

from __future__ import annotations

import logging
import threading
import time
import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify
from flask_socketio import SocketIO

from youtube_camera import YouTubeCameraSource, RegionConfig
from car_detector import CarDetector, Detection
from car_classifier import CarClassifier
from traffic_light_detector import TrafficLightDetector, LightColor
from car_tracker import CarTracker, EmbeddingExtractor
from crossroad_manager import CrossroadManager
from image_enhancer import ImageEnhancer
from deep_sort_tracker import DeepSortTracker
from anpr import ANPR

from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

app      = Flask(__name__)
socketio = SocketIO(app)

# ── Image enhancer ────────────────────────────────────────────────────────────

enhancer = ImageEnhancer(
    denoise    = "bilateral",   # "median" | "bilateral" | "wiener" | "none"
    clahe      = True,
    clahe_clip = 2.5,
    auto_clahe = True,          # dynamically adjusts clip per frame brightness
)

# ── YOLO vehicle + traffic light detectors ────────────────────────────────────

detector = CarDetector(
    model_size = "yolov5s",
    confidence = 0.45,
    save_crops = False,
    draw_boxes = False,
    frame_skip = 3,
)

tld = TrafficLightDetector(
    model_size = "yolov5s",
    confidence = 0.35,
    frame_skip = 3,
    min_pixels = 30,
)

# ── Car classifier (VMMR) ─────────────────────────────────────────────────────

classifier = CarClassifier(
    checkpoint_path    = os.getenv("CHECKPOINT_PATH"),
    confidence_threshold = 0.30,
    min_crop_size      = 48,
)

# ── ANPR ─────────────────────────────────────────────────────────────────────

anpr = ANPR(
    languages      = ["en"],
    min_confidence = 0.45,
    use_gpu        = True,
)

# ── DeepSORT tracker (per-region) ─────────────────────────────────────────────

_deep_sort: dict[str, DeepSortTracker] = {
    region: DeepSortTracker(
        max_age             = 30,
        n_init              = 3,
        max_cosine_distance = 0.4,
        embedder            = "mobilenet",
    )
    for region in ("top_left", "top_right", "bottom_left", "bottom_right")
}

# ── Cross-region car tracker + crossroad manager ──────────────────────────────

extractor = EmbeddingExtractor(classifier)
tracker   = CarTracker(extractor)
manager   = CrossroadManager(
    tracker      = tracker,
    set_phase_fn = lambda ns, ew, label: set_phase(ns, ew, label),
    add_log_fn   = lambda cam, event, t: add_log(cam, event, t),
)

# ── Annotated frame buffer ────────────────────────────────────────────────────

_annotated: dict[str, np.ndarray | None] = {
    "top_left": None, "top_right": None,
    "bottom_left": None, "bottom_right": None,
}
_annotated_lock = threading.Lock()

# ── Camera ────────────────────────────────────────────────────────────────────

cam = YouTubeCameraSource(
    url            = "https://www.youtube.com/watch?v=1H0iTzv2jiQ",
    name           = "intersection",
    fps_limit      = 15.0,
    reconnect_delay= 5.0,
    region_config  = RegionConfig(
        rects={
            # (x, y, width, height) in pixels — adjust to your stream
            "top_left":    (0,    0,    960,  540),   # North
            "top_right":   (960,  0,    960,  540),   # East
            "bottom_left": (0,    540,  960,  540),   # South
            "bottom_right":(960,  540,  960,  540),   # West
        }
    ),
)

REGION_DIR: dict[str, str] = {
    "top_left":     "north",
    "top_right":    "east",
    "bottom_left":  "south",
    "bottom_right": "west",
}
DIR_REGION: dict[str, str] = {v: k for k, v in REGION_DIR.items()}

_last_light_color: dict[str, str] = {}
_known_plates:     dict[str, str] = {}   # track_id -> plate


# ── Core detection handler ────────────────────────────────────────────────────

def _run_detection(region: str, raw_frame: np.ndarray) -> None:
    direction = REGION_DIR[region]
    ds        = _deep_sort[region]

    # ── 1. Image enhancement ──────────────────────────────────────────────
    frame = enhancer.process(raw_frame)

    # ── 2. YOLO vehicle detection ─────────────────────────────────────────
    detections, annotated = detector.detect_and_annotate(frame, direction=direction)

    # ── 3. DeepSORT tracking ──────────────────────────────────────────────
    ds_tracks = ds.update(detections, frame, direction=direction)

    # ── 4. Cross-region CarTracker update ────────────────────────────────
    # Build Detection-compatible list from DeepSORT tracks
    class _FakeDet:
        def __init__(self, t):
            self.bbox       = t.bbox
            self.confidence = t.confidence
            self.class_name = t.class_name
            self.crop       = t.crop

    fake_dets = [_FakeDet(t) for t in ds_tracks]
    crops     = [t.crop for t in ds_tracks]
    tracked   = tracker.update(direction, fake_dets, crops)

    # ── 5. Classify + ANPR per tracked car ───────────────────────────────
    for car, ds_track in zip(tracked, ds_tracks):
        crop = ds_track.crop

        # Enhanced crop for classifier + ANPR
        crop_enhanced = enhancer.enhance_crop(crop) if crop is not None else None

        # VMMR car model classification
        if crop_enhanced is not None:
            result = classifier.predict_crop(crop_enhanced)
            if result:
                model_name, conf = result
                tracker.update_model_name(car.id, model_name, conf)

        # ANPR — skip if already read plate for this track
        if crop_enhanced is not None and car.id not in _known_plates:
            plate = anpr.read(crop_enhanced)
            if plate:
                _known_plates[car.id] = plate.plate
                add_log(direction,
                        f"Plate: {plate.plate} [{car.short_id}] "
                        f"({plate.confidence:.0%})",
                        "success")
                logger.info("[%s] Plate: %s conf=%.2f",
                            direction, plate.plate, plate.confidence)

        # Draw ID + model + plate on annotated frame
        x1, y1, _, _ = car.bbox
        plate_txt = f" | {_known_plates[car.id]}" if car.id in _known_plates else ""
        label = f"[{car.short_id}] {car.model_name}{plate_txt}"
        cv2.putText(annotated, label, (x1, max(0, y1 - 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

        # Log new arrivals and direction changes
        if car.is_new:
            add_log(direction,
                    f"New: [{car.short_id}] {car.model_name} "
                    f"(w={car.priority_weight}x)",
                    "success")
        elif car.direction_changed:
            add_log(direction,
                    f"[{car.short_id}] moved "
                    f"{car.prev_direction} -> {direction}",
                    "info")

    # ── 6. Traffic light detection ────────────────────────────────────────
    tl_detections = tld.detect(frame, direction=direction)
    if tl_detections:
        tld.draw(annotated, tl_detections)
        best = max(tl_detections, key=lambda d: d.confidence * d.color_score)
        if best.color != LightColor.UNKNOWN:
            color_str = best.color.value
            if color_str != _last_light_color.get(direction):
                _last_light_color[direction] = color_str
                set_light(direction, color_str)
                add_log(direction,
                        f"Light: {color_str} ({best.confidence:.0%})",
                        "warn" if color_str == "yellow" else
                        "danger" if color_str == "red" else "success")

    # ── 7. Crossroad manager tick (Kalman-based) ──────────────────────────
    manager.tick()

    # ── 8. Queue overlay ──────────────────────────────────────────────────
    q      = tracker.count(direction)
    w_time = tracker.longest_wait(direction)
    kf     = manager.kalman_state()
    axis   = "ns" if direction in ("north", "south") else "ew"
    kf_score = kf.get(f"{axis}_smooth", 0.0)
    cv2.putText(
        annotated,
        f"Q:{q}  W:{w_time:.0f}s  KF:{kf_score:.1f}",
        (6, 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (0, 220, 255), 1, cv2.LINE_AA,
    )

    # ── 9. Push queues + push MJPEG buffer ───────────────────────────────
    push_queues()
    with _annotated_lock:
        _annotated[region] = annotated


# ── Region observers ──────────────────────────────────────────────────────────

def on_north(cam_name, region, frame): _run_detection("top_left",     frame)
def on_east (cam_name, region, frame): _run_detection("top_right",    frame)
def on_south(cam_name, region, frame): _run_detection("bottom_left",  frame)
def on_west (cam_name, region, frame): _run_detection("bottom_right", frame)
def on_full_frame(cam_name, full, regions): pass

cam.subscribe_region("top_left",     on_north)
cam.subscribe_region("top_right",    on_east)
cam.subscribe_region("bottom_left",  on_south)
cam.subscribe_region("bottom_right", on_west)
cam.subscribe_full(on_full_frame)


# ── MJPEG generator ───────────────────────────────────────────────────────────

def _annotated_mjpeg(region: str):
    while True:
        with _annotated_lock:
            frame = _annotated.get(region)
        if frame is None:
            time.sleep(0.05)
            continue
        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            time.sleep(0.05)
            continue
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            + buf.tobytes()
            + b"\r\n"
        )
        time.sleep(1.0 / cam.fps_limit)


# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video/<direction>")
def video(direction: str):
    region = DIR_REGION.get(direction)
    if region is None:
        return f"Unknown direction: {direction!r}", 404
    r = Response(_annotated_mjpeg(region),
                 mimetype="multipart/x-mixed-replace; boundary=frame")
    r.headers["X-Accel-Buffering"] = "no"
    r.headers["Cache-Control"]     = "no-cache"
    return r


@app.route("/video/full")
def video_full():
    return Response(cam.generate_mjpeg_full(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats")
def stats():
    phase = manager.current_phase()
    return jsonify({
        "camera":  cam.stats,
        "cars": [
            {
                "id":         c.id,
                "model":      c.model_name,
                "direction":  c.direction,
                "wait_s":     round(c.wait_time, 1),
                "frames":     c.frame_count,
                "confidence": round(c.confidence, 2),
                "weight":     c.priority_weight,
                "reason":     c.priority_reason,
                "plate":      _known_plates.get(c.id, ""),
            }
            for c in tracker.all_cars()
        ],
        "queues":  manager.queue_summary(),
        "scores":  manager.score_summary(),
        "kalman":  manager.kalman_state(),
        "plates":  dict(_known_plates),
        "phase": {
            "ns":       phase.ns,
            "ew":       phase.ew,
            "label":    phase.label,
            "ns_score": phase.ns_score,
            "ew_score": phase.ew_score,
        },
    })


@app.route("/snapshot")
def snapshot():
    ok = cam.debug_snapshot("debug_snapshot.jpg")
    return ("Saved" if ok else "No frame yet"), 200


# ── SocketIO ──────────────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    logger.info("Browser connected")


# ── Queue pusher ──────────────────────────────────────────────────────────────

_last_queue_push = 0.0

def push_queues() -> None:
    global _last_queue_push
    now = time.time()
    if now - _last_queue_push < 0.5:
        return
    _last_queue_push = now
    data = {}
    for direction in ("north", "south", "east", "west"):
        data[direction] = [
            {
                "id":     c.short_id,
                "model":  c.model_name,
                "wait":   int(c.wait_time),
                "weight": c.priority_weight,
                "reason": c.priority_reason,
                "plate":  _known_plates.get(c.id, ""),
            }
            for c in tracker.cars_in(direction)
        ]
    socketio.emit("update_queues", data)


# ── Dashboard helpers ─────────────────────────────────────────────────────────

def add_log(camera: str, event: str, type: str = "info") -> None:
    socketio.emit("add_log", {"camera": camera, "event": event, "type": type})

def set_light(direction: str, state: str) -> None:
    socketio.emit("set_light", {"direction": direction, "state": state})

def set_phase(ns: str, ew: str, label: str = "") -> None:
    socketio.emit("set_phase", {
        "ns":    ns,
        "ew":    ew,
        "label": label or f"N-S: {ns} / E-W: {ew}",
    })

def all_off() -> None:
    socketio.emit("all_off", {})

def flash_all(state: str, times: int = 5, interval_ms: int = 400) -> None:
    socketio.emit("flash_all", {"state": state, "times": times, "interval": interval_ms})

def set_stream(direction: str, url: str) -> None:
    socketio.emit("set_stream", {"dir": direction, "url": url})

def set_camera_offline(direction: str) -> None:
    socketio.emit("camera_offline", {"dir": direction})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cam.start()
    logger.info("http://0.0.0.0:5000")
    try:
        socketio.run(app, host="0.0.0.0", port=5000,
                     debug=False, allow_unsafe_werkzeug=True)
    finally:
        cam.stop()