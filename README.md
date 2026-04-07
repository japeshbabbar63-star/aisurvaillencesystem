# CityGuard AI — Smart Surveillance & Security System

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python: 3.8+](https://img.shields.io/badge/Python-3.8+-3776AB.svg?logo=python)
![YOLO: v8](https://img.shields.io/badge/Model-YOLO%20v8-00FFAA.svg)
![Flask: 3.0+](https://img.shields.io/badge/Framework-Flask%203.0+-black.svg?logo=flask)

CityGuard AI is a real-time, intelligent surveillance prototype designed for urban safety. It leverages computer vision and deep learning to identify suspicious scenarios automatically, reducing the need for constant human monitoring.

---

## ⚡ Key Features

- **Real-Time Detection**: Efficient object detection using a pretrained YOLOv8 nano model.
- **Suspicious Activity Rules**:
  - 🔴 **Crowd Clustering**: Detects when 3 or more people gather closely (potential fight/disturbance).
  - 🟠 **Unattended Objects**: Identifies bags (backpacks, suitcases) left without a person nearby.
  - 🟡 **Abnormal Motion**: Flags sudden, extreme movement spikes against a rolling history baseline.
- **Web Dashboard**: A premium, dark-themed Flask interface featuring:
  - Live annotated MJPEG video stream.
  - Real-time sliding alerts panel with severity indicators.
  - Persistent incident log with timestamped events.
- **Modular Architecture**: Clean separation between detection, analysis, and UI layers.

---

## 🛠️ Technology Stack

- **Core**: Python
- **Vision**: OpenCV & Ultralytics (YOLOv8)
- **Web**: Flask (Backend), HTML5/CSS3 (Frontend)
- **UI Design**: Modern Glassmorphism with custom animations

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/cityguard-ai.git
cd cityguard-ai
```

### 2. Install Dependencies
Ensure you have Python 3.8+ installed. Then run:
```bash
pip install -r requirements.txt
```

### 3. Run the Application
Start the surveillance server:
```bash
# To use your default webcam (Index 0)
python app.py

# To use a specific video file for testing
python app.py --source demo_video.mp4

# To run on a different port
python app.py --port 8080
```

Once running, navigate to **`http://localhost:5000`** in your web browser.

---

## 📁 Project Structure

```text
├── analysis/
│   └── analyzer.py      # Rule-based logic for suspicious activity
├── detection/
│   └── detector.py      # YOLOv8 inference and frame annotation
├── static/
│   └── style.css        # Dashboard styling (dark theme, glassmorphism)
├── templates/
│   └── index.html       # Web dashboard template
├── app.py               # Main Flask application entry point
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation (this file)
```

---

## ⚙️ Configuration

You can adjust detection thresholds in `app.py` or directly in the module initializers:
- `crowd_distance`: Max pixels between people in a cluster (Default: 120).
- `unattended_distance`: Max pixels between a bag and a person (Default: 160).
- `motion_threshold`: Sensitivity for abnormal motion spikes (Default: 3.5x).

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

*Built for Smart City Safety & Security hackathons.*
