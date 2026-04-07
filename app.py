"""
Smart City Surveillance System — Flask Application
====================================================
Entry point for the web-based surveillance dashboard.
Serves annotated MJPEG video stream and real-time alert data.

Usage:
    python app.py                    # Use default webcam
    python app.py --source video.mp4 # Use a video file
    python app.py --source 0         # Webcam index
"""

import argparse
import threading
import time
import json
from datetime import datetime

import cv2
from flask import Flask, Response, render_template, jsonify

from detection.detector import ObjectDetector
from analysis.analyzer import SuspiciousActivityAnalyzer

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

# Shared state (thread-safe via GIL for simple reads/writes)
latest_frame = None
latest_alerts: list[dict] = []
incident_log: list[dict] = []
system_status = {"fps": 0, "objects": 0, "running": False}

# Lock for frame access
frame_lock = threading.Lock()

# Max incidents to keep in memory
MAX_INCIDENTS = 200


# ---------------------------------------------------------------------------
# Video processing thread
# ---------------------------------------------------------------------------
def video_processing_loop(source):
    """
    Continuously capture frames, run detection + analysis, and update shared state.

    Args:
        source: Video source — integer for webcam index, or string path to video file.
    """
    global latest_frame, latest_alerts, incident_log, system_status

    # Initialize detector and analyzer
    detector = ObjectDetector(model_path="yolov8n.pt", confidence=0.40)
    analyzer = SuspiciousActivityAnalyzer(
        crowd_distance=120,
        crowd_min_people=3,
        unattended_distance=160,
        motion_threshold=3.5,
    )

    # Open video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        return

    system_status["running"] = True
    print(f"[INFO] Video capture started — source: {source}")

    fps_counter = 0
    fps_timer = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Loop video files for continuous demo
            if isinstance(source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        # Resize for performance (max width 720px)
        h, w = frame.shape[:2]
        if w > 720:
            scale = 720 / w
            frame = cv2.resize(frame, (720, int(h * scale)))

        # --- Detection ---
        detections = detector.detect(frame)
        annotated = detector.annotate(frame, detections)

        # --- Analysis ---
        alerts = analyzer.analyze(frame, detections)
        annotated = analyzer.draw_alerts(annotated, alerts)

        # --- Draw status bar ---
        cv2.rectangle(annotated, (0, annotated.shape[0] - 28),
                      (annotated.shape[1], annotated.shape[0]), (30, 30, 30), -1)
        status_text = (
            f"Objects: {len(detections)}  |  "
            f"Alerts: {len(alerts)}  |  "
            f"FPS: {system_status['fps']}  |  "
            f"{datetime.now().strftime('%H:%M:%S')}"
        )
        cv2.putText(annotated, status_text, (10, annotated.shape[0] - 8),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # --- Update shared state ---
        with frame_lock:
            latest_frame = annotated.copy()

        latest_alerts = alerts
        system_status["objects"] = len(detections)

        # Log new incidents
        for alert in alerts:
            incident = {
                "id": len(incident_log) + 1,
                "type": alert["type"],
                "message": alert["message"],
                "severity": alert["severity"],
                "timestamp": alert["timestamp"],
            }
            incident_log.insert(0, incident)
            if len(incident_log) > MAX_INCIDENTS:
                incident_log.pop()

        # FPS calculation
        fps_counter += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            system_status["fps"] = round(fps_counter / elapsed)
            fps_counter = 0
            fps_timer = time.time()

    cap.release()
    system_status["running"] = False
    print("[INFO] Video capture stopped.")


def generate_mjpeg():
    """Generator that yields MJPEG frames for the /video_feed endpoint."""
    while True:
        with frame_lock:
            frame = latest_frame

        if frame is None:
            time.sleep(0.05)
            continue

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
        )
        time.sleep(0.03)  # ~30 fps cap


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Serve the main dashboard page."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG stream of annotated video frames."""
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/alerts")
def get_alerts():
    """Return current frame alerts as JSON."""
    return jsonify({
        "alerts": latest_alerts,
        "status": system_status,
    })


@app.route("/incidents")
def get_incidents():
    """Return full incident log as JSON."""
    return jsonify({
        "incidents": incident_log[:50],  # Latest 50
        "total": len(incident_log),
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart City Surveillance System")
    parser.add_argument(
        "--source", default="0",
        help="Video source: webcam index (0, 1, ...) or path to video file",
    )
    parser.add_argument("--port", type=int, default=5000, help="Flask server port")
    args = parser.parse_args()

    # Parse source: integer for webcam, string for file path
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    # Start video processing in background thread
    video_thread = threading.Thread(target=video_processing_loop, args=(source,), daemon=True)
    video_thread.start()

    print(f"[INFO] Dashboard running at http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
