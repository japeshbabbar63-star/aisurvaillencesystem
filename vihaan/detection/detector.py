"""
Detection Module — YOLOv8 Inference Wrapper
============================================
Uses ultralytics YOLOv8 nano model (pretrained) for real-time object detection.
Filters for security-relevant classes: person, backpack, handbag, suitcase.
"""

from ultralytics import YOLO
import cv2
import numpy as np

# COCO class IDs we care about
RELEVANT_CLASSES = {
    0: "person",
    24: "backpack",
    26: "handbag",
    28: "suitcase",
}

# Color palette for bounding boxes (BGR)
COLORS = {
    "person": (0, 220, 100),      # green
    "backpack": (0, 165, 255),    # orange
    "handbag": (0, 165, 255),     # orange
    "suitcase": (0, 165, 255),    # orange
}


class ObjectDetector:
    """Wraps YOLOv8 for security-focused object detection."""

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.40):
        """
        Initialize the detector.

        Args:
            model_path: Path or name of the YOLO model (auto-downloads if needed).
            confidence: Minimum confidence threshold for detections.
        """
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run YOLOv8 inference on a single frame.

        Args:
            frame: BGR image (numpy array from OpenCV).

        Returns:
            List of detection dicts with keys:
                - label (str): class name
                - confidence (float): detection confidence
                - bbox (tuple): (x1, y1, x2, y2)
                - center (tuple): (cx, cy)
        """
        results = self.model(frame, conf=self.confidence, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id not in RELEVANT_CLASSES:
                    continue

                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                detections.append({
                    "label": RELEVANT_CLASSES[cls_id],
                    "confidence": round(conf, 2),
                    "bbox": (x1, y1, x2, y2),
                    "center": (cx, cy),
                })

        return detections

    def annotate(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on a frame.

        Args:
            frame: Original BGR frame.
            detections: List of detection dicts from detect().

        Returns:
            Annotated frame copy.
        """
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            conf = det["confidence"]
            color = COLORS.get(label, (200, 200, 200))

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label background
            text = f'{label} {conf:.0%}'
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return annotated
