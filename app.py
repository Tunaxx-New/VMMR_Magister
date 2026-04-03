"""
app.py - Crossroads monitor — full pipeline.
All configuration loaded from .env via config.py.

Frame pipeline per region:
    YouTube stream
        -> ImageEnhancer  (Bilateral/Median/Wiener + adaptive CLAHE)
        -> YOLOv5         (vehicle + traffic light detection)
        -> DeepSORT       (multi-object tracking with Kalman + re-ID)
        -> CarTracker     (unique IDs, cross-region re-ID, priority weights)
        -> CarClassifier  (VMMR model name + temporal vote accumulator)
        -> ANPR           (licence plate OCR)
        -> CrossroadManager (Kalman-filtered priority scores, auto phase switching)
        -> Dashboard      (MJPEG stream + SocketIO events)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter

import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify
from flask_socketio import SocketIO

from config import cfg                                   # ← all values from .env
from youtube_camera import YouTubeCameraSource, RegionConfig
from car_detector import CarDetector
from car_classifier import CarClassifier
from traffic_light_detector import TrafficLightDetector, LightColor
from car_tracker import CarTracker, EmbeddingExtractor
from crossroad_manager import CrossroadManager
from image_enhancer import ImageEnhancer
from deep_sort_tracker import DeepSortTracker
from anpr import ANPR

logging.basicConfig(
    level=getattr(logging, cfg.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

app      = Flask(__name__)
socketio = SocketIO(app)

# ── Image enhancer ────────────────────────────────────────────────────────────

enhancer = ImageEnhancer(
    denoise        = cfg.DENOISE_METHOD,
    clahe          = cfg.CLAHE_ENABLED,
    clahe_clip     = cfg.CLAHE_CLIP,
    auto_clahe     = cfg.CLAHE_AUTO,
    bilateral_d    = cfg.BILATERAL_D,
    bilateral_sigma= cfg.BILATERAL_SIGMA,
    median_ksize   = cfg.MEDIAN_KSIZE,
)

# ── YOLO vehicle + traffic light detectors ────────────────────────────────────

detector = CarDetector(
    model_size    = cfg.YOLO_MODEL,
    confidence    = cfg.YOLO_CONFIDENCE,
    iou_threshold = cfg.YOLO_IOU,
    save_crops    = False,
    draw_boxes    = False,
    frame_skip    = cfg.YOLO_FRAME_SKIP,
)

tld = TrafficLightDetector(
    model_size  = cfg.YOLO_MODEL,
    confidence  = cfg.TLD_CONFIDENCE,
    frame_skip  = cfg.TLD_FRAME_SKIP,
    min_pixels  = cfg.TLD_MIN_PIXELS,
)

# ── Car classifier (VMMR) ─────────────────────────────────────────────────────

classifier = CarClassifier(
    checkpoint_path      = cfg.CLASSIFIER_CHECKPOINT,
    confidence_threshold = cfg.CLASSIFIER_CONFIDENCE,
    min_crop_size        = cfg.CLASSIFIER_MIN_CROP,
)

# ── ANPR ──────────────────────────────────────────────────────────────────────

anpr = ANPR(
    languages      = cfg.ANPR_LANGUAGES,
    min_confidence = cfg.ANPR_CONFIDENCE,
    use_gpu        = cfg.ANPR_GPU,
) if cfg.ANPR_ENABLED else None

# ── DeepSORT tracker (one per region) ─────────────────────────────────────────

_deep_sort: dict[str, DeepSortTracker] = {
    region: DeepSortTracker(
        max_age             = cfg.DEEPSORT_MAX_AGE,
        n_init              = cfg.DEEPSORT_N_INIT,
        max_cosine_distance = cfg.DEEPSORT_MAX_DIST,
        embedder            = cfg.DEEPSORT_EMBEDDER,
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
    url             = cfg.YOUTUBE_URL,
    name            = cfg.CAM_NAME,
    fps_limit       = cfg.CAM_FPS_LIMIT,
    reconnect_delay = cfg.CAM_RECONNECT_DELAY,
    region_config   = RegionConfig(
        rects={
            "top_left":    cfg.RECT_NORTH,
            "top_right":   cfg.RECT_EAST,
            "bottom_left": cfg.RECT_SOUTH,
            "bottom_right":cfg.RECT_WEST,
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
_known_plates:     dict[str, str] = {}   # car.id -> plate text


# ── Label drawing helper ──────────────────────────────────────────────────────

def _draw_label(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    id_line: str,
    model_line: str,
    priority_weight: float = 1.0,
) -> None:
    """Two-line coloured label box above each vehicle bbox."""
    if priority_weight >= 4.0:
        box_color = (0,   0,   220)
        txt_color = (255, 255, 255)
    elif priority_weight >= 2.0:
        box_color = (0,   140, 255)
        txt_color = (255, 255, 255)
    elif priority_weight <= 0.8:
        box_color = (220, 200,   0)
        txt_color = (0,     0,   0)
    else:
        box_color = (200, 200, 200)
        txt_color = (0,     0,   0)

    font  = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.42
    thick = 1
    pad   = 3

    (w1, h1), _ = cv2.getTextSize(id_line,    font, scale, thick)
    (w2, h2), _ = cv2.getTextSize(model_line, font, scale, thick)

    box_w = max(w1, w2) + pad * 2
    box_h = h1 + h2 + pad * 3
    lx    = x1
    ly    = max(0, y1 - box_h - 2)
    if ly == 0:
        ly = y1 + 2

    cv2.rectangle(frame, (lx, ly), (lx + box_w, ly + box_h), box_color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
    cv2.putText(frame, id_line,
                (lx + pad, ly + h1 + pad),
                font, scale, txt_color, thick, cv2.LINE_AA)
    cv2.putText(frame, model_line,
                (lx + pad, ly + h1 + h2 + pad * 2),
                font, scale, txt_color, thick, cv2.LINE_AA)


# ── Model vote accumulator ────────────────────────────────────────────────────

_model_votes: dict[str, Counter] = {}


_YOLO_CLASSES = {"car", "truck", "bus", "motorcycle", "vehicle"}

def _cast_vote(car_id: str, model_name: str, confidence: float) -> str | None:
    """Weighted temporal voting — returns stable winner or None.
    Ignores YOLO raw class names so they never appear as model labels.
    """
    if not model_name or model_name.lower().strip() in _YOLO_CLASSES:
        return None
    if car_id not in _model_votes:
        _model_votes[car_id] = Counter()

    weight = max(1, int(round(confidence * 10)))
    _model_votes[car_id][model_name] += weight

    ctr = _model_votes[car_id]
    if sum(ctr.values()) > cfg.VOTE_WINDOW * 5:
        for k in list(ctr.keys()):
            ctr[k] = max(0, ctr[k] - 1)
            if ctr[k] == 0:
                del ctr[k]

    if not ctr:
        return None
    winner, top_count = ctr.most_common(1)[0]
    return winner if top_count >= cfg.VOTE_MIN_WIN else None


# ── Core detection handler ────────────────────────────────────────────────────

def _run_detection(region: str, raw_frame: np.ndarray) -> None:
    direction = REGION_DIR[region]
    ds        = _deep_sort[region]

    # 1. Image enhancement
    frame = enhancer.process(raw_frame)

    # 2. YOLO vehicle detection
    detections, annotated = detector.detect_and_annotate(frame, direction=direction)

    # 3. DeepSORT tracking
    ds_tracks = ds.update(detections, frame, direction=direction)

    # 4. Cross-region CarTracker update
    class _FakeDet:
        def __init__(self, t):
            self.bbox       = t.bbox
            self.confidence = t.confidence
            self.class_name = t.class_name
            self.crop       = t.crop

    tracked = tracker.update(direction, [_FakeDet(t) for t in ds_tracks],
                              [t.crop for t in ds_tracks])

    # 5. Classify + ANPR + draw per tracked car
    # Build a map from DeepSORT track_id to car so we can match on skipped frames
    ds_by_tid = {t.track_id: t for t in ds_tracks}

    for car, ds_track in zip(tracked, ds_tracks):
        crop          = ds_track.crop
        crop_enhanced = enhancer.enhance_crop(crop) if crop is not None else None

        # VMMR classification with TTA for mature tracks
        if crop_enhanced is not None:
            use_tta = (
                cfg.CLASSIFIER_TTA_AUGMENTS > 1
                and car.frame_count >= cfg.CLASSIFIER_TTA_MIN_FRAMES
            )
            result = (
                classifier.predict_with_tta(crop_enhanced, cfg.CLASSIFIER_TTA_AUGMENTS)
                if use_tta
                else classifier.predict_crop(crop_enhanced)
            )
            if result:
                model_name, conf = result
                voted = _cast_vote(car.id, model_name, conf)
                if voted:
                    tracker.update_model_name(car.id, voted, conf)

        # ANPR — once per car
        if anpr and crop_enhanced is not None and car.id not in _known_plates:
            plate = anpr.read(crop_enhanced)
            if plate:
                _known_plates[car.id] = plate.plate
                add_log(direction,
                        f"Plate: {plate.plate} [{car.short_id}] ({plate.confidence:.0%})",
                        "success")

        # Draw box + ID label — this is the ONLY place any text is drawn
        # YOLO's _draw is disabled so no "car 87%" labels appear
        x1, y1, x2, y2 = car.bbox
        plate_txt  = f"  {_known_plates[car.id]}" if car.id in _known_plates else ""
        id_line    = f"#{ds_track.track_id}  [{car.short_id}]"
        model_line = f"{car.model_name}{plate_txt}"
        _draw_label(annotated, x1, y1, x2, y2, id_line, model_line,
                    car.priority_weight)

        # Log new / moved cars
        if car.is_new:
            add_log(direction,
                    f"New: [{car.short_id}] {car.model_name} (w={car.priority_weight}x)",
                    "success")
        elif car.direction_changed:
            add_log(direction,
                    f"[{car.short_id}] moved {car.prev_direction} -> {direction}",
                    "info")

    # On YOLO-skipped frames DeepSORT still returns tracks with predicted bboxes
    # — draw any tracked cars that didn't appear in the tracker list this frame
    tracked_ids = {car.id for car in tracked}
    for car in tracker.cars_in(direction):
        if car.id not in tracked_ids and car.lost_frames < 3:
            x1, y1, x2, y2 = car.bbox
            plate_txt  = f"  {_known_plates[car.id]}" if car.id in _known_plates else ""
            id_line    = f"[{car.short_id}] (pred)"
            model_line = f"{car.model_name}{plate_txt}"
            _draw_label(annotated, x1, y1, x2, y2, id_line, model_line,
                        car.priority_weight)

    # 6. Traffic light detection
    tl_detections = tld.detect(frame, direction=direction)
    if tl_detections:
        tld.draw(annotated, tl_detections)
        best = max(tl_detections, key=lambda d: d.confidence * d.color_score)
        if best.color != LightColor.UNKNOWN:
            color_str = best.color.value
            if color_str != _last_light_color.get(direction):
                _last_light_color[direction] = color_str
                set_light(direction, color_str)
                add_log(direction, f"Light: {color_str} ({best.confidence:.0%})",
                        "warn" if color_str == "yellow" else
                        "danger" if color_str == "red" else "success")

    # 7. Crossroad manager tick (Kalman-based)
    manager.tick()

    # 8. Queue overlay
    q      = tracker.count(direction)
    w_time = tracker.longest_wait(direction)
    kf     = manager.kalman_state()
    axis   = "ns" if direction in ("north", "south") else "ew"
    kf_score = kf.get(f"{axis}_smooth", 0.0)
    cv2.putText(annotated,
                f"Q:{q}  W:{w_time:.0f}s  KF:{kf_score:.1f}",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 220, 255), 1, cv2.LINE_AA)

    # 9. Push to dashboard
    push_queues()
    push_kalman()
    with _annotated_lock:
        _annotated[region] = annotated


# ── Region observers ──────────────────────────────────────────────────────────

def on_north(cn, r, f): _run_detection("top_left",     f)
def on_east (cn, r, f): _run_detection("top_right",    f)
def on_south(cn, r, f): _run_detection("bottom_left",  f)
def on_west (cn, r, f): _run_detection("bottom_right", f)
def on_full (cn, full, regions): pass

cam.subscribe_region("top_left",     on_north)
cam.subscribe_region("top_right",    on_east)
cam.subscribe_region("bottom_left",  on_south)
cam.subscribe_region("bottom_right", on_west)
cam.subscribe_full(on_full)


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
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")
        time.sleep(1.0 / cfg.CAM_FPS_LIMIT)


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
        "cars": [{
            "id":         c.id,
            "model":      c.model_name,
            "direction":  c.direction,
            "wait_s":     round(c.wait_time, 1),
            "frames":     c.frame_count,
            "confidence": round(c.confidence, 2),
            "weight":     c.priority_weight,
            "reason":     c.priority_reason,
            "plate":      _known_plates.get(c.id, ""),
        } for c in tracker.all_cars()],
        "queues":  manager.queue_summary(),
        "scores":  manager.score_summary(),
        "kalman":  manager.kalman_state(),
        "plates":  dict(_known_plates),
        "phase":   {"ns": phase.ns, "ew": phase.ew, "label": phase.label,
                    "ns_score": phase.ns_score, "ew_score": phase.ew_score},
    })

@app.route("/config")
def config_view():
    """Show all active configuration values."""
    return jsonify(cfg.dump())

@app.route("/snapshot")
def snapshot():
    ok = cam.debug_snapshot("debug_snapshot.jpg")
    return ("Saved" if ok else "No frame yet"), 200


# ── SocketIO ──────────────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    logger.info("Browser connected")


# ── Dashboard pushers ─────────────────────────────────────────────────────────

_last_queue_push  = 0.0
_last_kalman_push = 0.0


def push_queues() -> None:
    global _last_queue_push
    now = time.time()
    if now - _last_queue_push < 0.5:
        return
    _last_queue_push = now
    data = {}
    for direction in ("north", "south", "east", "west"):
        data[direction] = [{
            "id":     c.short_id,
            "model":  c.model_name,
            "wait":   int(c.wait_time),
            "weight": c.priority_weight,
            "reason": c.priority_reason,
            "plate":  _known_plates.get(c.id, ""),
        } for c in tracker.cars_in(direction)]
    socketio.emit("update_queues", data)


def push_kalman() -> None:
    global _last_kalman_push
    now = time.time()
    if now - _last_kalman_push < 1.0:
        return
    _last_kalman_push = now
    kf = manager.kalman_state()
    if kf:
        socketio.emit("kalman_state", kf)


# ── Dashboard helpers ─────────────────────────────────────────────────────────

def add_log(camera: str, event: str, type: str = "info") -> None:
    socketio.emit("add_log", {"camera": camera, "event": event, "type": type})

def set_light(direction: str, state: str) -> None:
    socketio.emit("set_light", {"direction": direction, "state": state})

def set_phase(ns: str, ew: str, label: str = "") -> None:
    socketio.emit("set_phase", {"ns": ns, "ew": ew,
                                "label": label or f"N-S: {ns} / E-W: {ew}"})

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
    logger.info("Config loaded — %d keys", len(cfg.dump()))
    logger.info("Stream: %s", cfg.YOUTUBE_URL)
    cam.start()
    logger.info("Running at http://%s:%d", cfg.HOST, cfg.PORT)
    try:
        socketio.run(app, host=cfg.HOST, port=cfg.PORT,
                     debug=cfg.DEBUG, allow_unsafe_werkzeug=True)
    finally:
        cam.stop()