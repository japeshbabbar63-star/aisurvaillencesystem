"""
Analysis Module — Suspicious Activity Detection
=================================================
Rule-based logic layer that analyzes YOLOv8 detections to identify:
  1. Crowd clustering (potential fights/disturbances)
  2. Unattended objects (bags without nearby persons)
  3. Abnormal motion (sudden movement spikes)
"""

import math
import time
import cv2
import numpy as np
from datetime import datetime


class SuspiciousActivityAnalyzer:
    """Analyzes detections for suspicious activity patterns."""

    def __init__(
        self,
        crowd_distance: int = 100,
        crowd_min_people: int = 3,
        unattended_distance: int = 150,
        motion_threshold: float = 3.0,
        motion_history_size: int = 15,
    ):
        """
        Configure detection thresholds.

        Args:
            crowd_distance: Max pixel distance between people to count as clustered.
            crowd_min_people: Minimum people in a cluster to trigger alert.
            unattended_distance: Max pixel distance from bag to nearest person.
            motion_threshold: Multiplier over average motion to flag as abnormal.
            motion_history_size: Number of frames to average for motion baseline.
        """
        self.crowd_distance = crowd_distance
        self.crowd_min_people = crowd_min_people
        self.unattended_distance = unattended_distance
        self.motion_threshold = motion_threshold
        self.motion_history_size = motion_history_size

        # Internal state
        self._prev_gray = None
        self._motion_history: list[float] = []

    @staticmethod
    def _distance(p1: tuple, p2: tuple) -> float:
        """Euclidean distance between two (x, y) points."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    # ------------------------------------------------------------------
    # Rule 1: Crowd clustering
    # ------------------------------------------------------------------
    def _check_crowd_clustering(self, detections: list[dict]) -> list[dict]:
        """Detect groups of people standing too close together."""
        persons = [d for d in detections if d["label"] == "person"]
        alerts = []

        if len(persons) < self.crowd_min_people:
            return alerts

        # Build adjacency: which persons are within crowd_distance of each other
        n = len(persons)
        visited = [False] * n

        def _bfs(start: int) -> list[int]:
            """Find connected cluster via BFS."""
            queue = [start]
            cluster = []
            visited[start] = True
            while queue:
                idx = queue.pop(0)
                cluster.append(idx)
                for j in range(n):
                    if not visited[j] and self._distance(
                        persons[idx]["center"], persons[j]["center"]
                    ) < self.crowd_distance:
                        visited[j] = True
                        queue.append(j)
            return cluster

        for i in range(n):
            if visited[i]:
                continue
            cluster = _bfs(i)
            if len(cluster) >= self.crowd_min_people:
                # Average center of cluster for annotation
                cx = int(np.mean([persons[k]["center"][0] for k in cluster]))
                cy = int(np.mean([persons[k]["center"][1] for k in cluster]))
                alerts.append({
                    "type": "Crowd Cluster",
                    "message": f"{len(cluster)} people clustered — possible disturbance",
                    "severity": "high",
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "location": (cx, cy),
                })

        return alerts

    # ------------------------------------------------------------------
    # Rule 2: Unattended objects
    # ------------------------------------------------------------------
    def _check_unattended_objects(self, detections: list[dict]) -> list[dict]:
        """Detect bags/suitcases with no person nearby."""
        bag_labels = {"backpack", "handbag", "suitcase"}
        bags = [d for d in detections if d["label"] in bag_labels]
        persons = [d for d in detections if d["label"] == "person"]
        alerts = []

        for bag in bags:
            nearest = float("inf")
            for person in persons:
                dist = self._distance(bag["center"], person["center"])
                nearest = min(nearest, dist)

            if nearest > self.unattended_distance:
                alerts.append({
                    "type": "Unattended Object",
                    "message": f"Unattended {bag['label']} detected — no person nearby",
                    "severity": "medium",
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "location": bag["center"],
                })

        return alerts

    # ------------------------------------------------------------------
    # Rule 3: Abnormal motion
    # ------------------------------------------------------------------
    def _check_abnormal_motion(self, frame: np.ndarray) -> list[dict]:
        """Detect sudden spikes in overall frame motion."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        alerts = []

        if self._prev_gray is not None:
            delta = cv2.absdiff(self._prev_gray, gray)
            thresh = cv2.threshold(delta, 30, 255, cv2.THRESH_BINARY)[1]
            motion_score = float(np.sum(thresh) / 255)

            self._motion_history.append(motion_score)
            if len(self._motion_history) > self.motion_history_size:
                self._motion_history.pop(0)

            if len(self._motion_history) >= 5:
                avg_motion = np.mean(self._motion_history[:-1])
                if avg_motion > 0 and motion_score > avg_motion * self.motion_threshold:
                    alerts.append({
                        "type": "Abnormal Motion",
                        "message": f"Sudden motion spike detected ({motion_score:.0f} vs avg {avg_motion:.0f})",
                        "severity": "low",
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "location": None,
                    })

        self._prev_gray = gray
        return alerts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(self, frame: np.ndarray, detections: list[dict]) -> list[dict]:
        """
        Run all suspicious activity checks.

        Args:
            frame: Current BGR frame.
            detections: List of detection dicts from ObjectDetector.detect().

        Returns:
            List of alert dicts with keys: type, message, severity, timestamp, location.
        """
        alerts = []
        alerts.extend(self._check_crowd_clustering(detections))
        alerts.extend(self._check_unattended_objects(detections))
        alerts.extend(self._check_abnormal_motion(frame))
        return alerts

    def draw_alerts(self, frame: np.ndarray, alerts: list[dict]) -> np.ndarray:
        """
        Draw alert indicators on the frame.

        Args:
            frame: Annotated frame (already has bounding boxes).
            alerts: List of alert dicts.

        Returns:
            Frame with alert overlays.
        """
        severity_colors = {
            "high": (0, 0, 255),       # red
            "medium": (0, 140, 255),   # orange
            "low": (0, 255, 255),      # yellow
        }

        for alert in alerts:
            color = severity_colors.get(alert["severity"], (255, 255, 255))
            loc = alert.get("location")

            if loc:
                # Draw pulsing circle at alert location
                cv2.circle(frame, loc, 40, color, 3)
                cv2.circle(frame, loc, 45, color, 1)

            # Draw alert banner at top
            text = f"⚠ {alert['type']}: {alert['message']}"
            y_offset = 30 + alerts.index(alert) * 30
            cv2.rectangle(frame, (0, y_offset - 20), (len(text) * 9 + 10, y_offset + 5), color, -1)
            cv2.putText(frame, text, (5, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return frame
