"""
app.py - Crossroads monitor: YouTube stream -> 4 regions -> YOLOv5 -> dashboard.

Cars are detected and bounding boxes drawn directly on the MJPEG streams.
No files are saved to disk.
"""

from __future__ import annotations

import logging
import threading
import numpy as np
from flask import Flask, Response, render_template, jsonify
from flask_socketio import SocketIO

from youtube_camera import YouTubeCameraSource, RegionConfig
from car_detector import CarDetector, Detection
from car_classifier import CarClassifier
from traffic_light_detector import TrafficLightDetector, LightColor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

app      = Flask(__name__)
socketio = SocketIO(app)

# ── YOLOv5 detector ───────────────────────────────────────────────────────────

detector = CarDetector(
    model_size="yolov5s",
    confidence=0.45,
    save_crops=False,      # no files saved
    draw_boxes=False,      # we draw manually below so we control timing
    frame_skip=3,
)

# ── VMMR car model classifier ─────────────────────────────────────────────────

classifier = CarClassifier(
    checkpoint_path="clip_cars_checkpoint.pth",  # <- your checkpoint
    confidence_threshold=0.30,
    min_crop_size=48,
)

# ── Traffic light detector ────────────────────────────────────────────────────

tld = TrafficLightDetector(
    model_size="yolov5s",   # reuses same weights as car detector
    confidence=0.35,
    frame_skip=3,
    min_pixels=30,
)

# Track last seen color per direction to avoid spamming identical commands
_last_light_color: dict[str, str] = {}

# ── Annotated frame buffers (region -> latest BGR frame with boxes drawn) ─────
#
# The camera's _notify() stores raw frames. We run YOLO in the observer
# callbacks, draw boxes, then store the result here. The MJPEG generators
# read from this dict instead of the raw camera buffer.

_annotated: dict[str, np.ndarray | None] = {
    "top_left":     None,
    "top_right":    None,
    "bottom_left":  None,
    "bottom_right": None,
}
_annotated_lock = threading.Lock()

# ── Camera ────────────────────────────────────────────────────────────────────

cam = YouTubeCameraSource(
    url="https://www.youtube.com/watch?v=1H0iTzv2jiQ",
    name="intersection",
    fps_limit=15.0,
    region_config=RegionConfig(
        rects={
            #              x,    y,    w,    h
            "top_left":   (0,    0,    1000,  300),   # North
            "top_right":  (0,  180,    400,  500),   # East
            "bottom_left":(500,    500,  800,  600),   # South
            "bottom_right":(900, 200,  1280,  400),   # West
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


# ── Shared detection handler ──────────────────────────────────────────────────

def _run_detection(region: str, frame: np.ndarray) -> None:
    """
    1. YOLOv5 -> vehicle detections + annotated frame.
    2. Each vehicle crop -> VMMR classifier -> car model name -> log.
    3. YOLOv5 -> traffic light detections -> HSV color read.
    4. If color changed -> emit set_light command to dashboard.
    5. Push annotated frame to MJPEG buffer.
    """
    direction = REGION_DIR[region]

    # ── Vehicle detection + classification ───────────────────────────────
    detections, annotated = detector.detect_and_annotate(frame, direction=direction)

    for det in detections:
        if det.crop is not None:
            result = classifier.predict_crop(det.crop)
            if result:
                car_model, conf = result
                event    = f"{car_model} ({conf:.0%})"
                log_type = "success"
            else:
                event    = f"{det.class_name.capitalize()} detected ({det.confidence:.0%})"
                log_type = "info"
        else:
            event    = f"{det.class_name.capitalize()} detected ({det.confidence:.0%})"
            log_type = "info"

        add_log(direction, event, log_type)
        logger.info("[%s] %s", direction, event)

    # ── Traffic light detection ───────────────────────────────────────────
    tl_detections = tld.detect(frame, direction=direction)

    if tl_detections:
        # Draw traffic light boxes on the annotated frame
        tld.draw(annotated, tl_detections)

        # Pick the most confident detection
        best = max(tl_detections, key=lambda d: d.confidence * d.color_score)

        if best.color != LightColor.UNKNOWN:
            color_str = best.color.value   # "red" | "yellow" | "green"
            prev      = _last_light_color.get(direction)

            if color_str != prev:
                # Color changed — update dashboard light and log
                _last_light_color[direction] = color_str
                set_light(direction, color_str)
                add_log(
                    direction,
                    f"Traffic light: {color_str} ({best.confidence:.0%})",
                    "warn" if color_str == "yellow" else
                    "danger" if color_str == "red" else "success",
                )
                logger.info("[%s] Traffic light -> %s", direction, color_str)

    with _annotated_lock:
        _annotated[region] = annotated


# ── Region observer callbacks ─────────────────────────────────────────────────

def on_north(cam_name: str, region: str, frame: np.ndarray) -> None:
    _run_detection("top_left", frame)


def on_east(cam_name: str, region: str, frame: np.ndarray) -> None:
    _run_detection("top_right", frame)


def on_south(cam_name: str, region: str, frame: np.ndarray) -> None:
    _run_detection("bottom_left", frame)


def on_west(cam_name: str, region: str, frame: np.ndarray) -> None:
    _run_detection("bottom_right", frame)


def on_full_frame(
    cam_name: str,
    full: np.ndarray,
    regions: dict[str, np.ndarray],
) -> None:
    pass


# Wire subscriptions
cam.subscribe_region("top_left",     on_north)
cam.subscribe_region("top_right",    on_east)
cam.subscribe_region("bottom_left",  on_south)
cam.subscribe_region("bottom_right", on_west)
cam.subscribe_full(on_full_frame)


# ── MJPEG generators (read from annotated buffer) ────────────────────────────

import cv2
import time

def _annotated_mjpeg(region: str):
    """Serve annotated frames (with YOLO boxes) for a given region."""
    while True:
        with _annotated_lock:
            frame = _annotated.get(region)

        if frame is None:
            # No frame yet — wait briefly
            time.sleep(0.05)
            continue

        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            time.sleep(0.05)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
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
    return Response(
        _annotated_mjpeg(region),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/video/full")
def video_full():
    return Response(
        cam.generate_mjpeg_full(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/stats")
def stats():
    return jsonify(cam.stats)


@app.route("/snapshot")
def snapshot():
    ok = cam.debug_snapshot("debug_snapshot.jpg")
    return ("Snapshot saved" if ok else "No frame yet"), 200


# ── SocketIO ──────────────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    logger.info("Browser connected")


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
    logger.info("Camera started - http://127.0.0.1:5000")
    try:
        socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
    finally:
        cam.stop()