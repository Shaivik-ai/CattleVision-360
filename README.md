<div align="center">

# 🐄 CattleVision 360: Biometric Muzzle Kiosk & RAG Health Diagnostics

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-005571?logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-yellow)](https://github.com/facebookresearch/faiss)
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-00FFFF)](https://ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

*An end-to-end computer vision and RAG veterinary diagnostic system leveraging bovine rhinoglyphics (muzzle prints), deep metric embeddings, FAISS vector search, and automated health anomaly alerts.*

</div>

---

## 🌟 Executive Summary

**CattleVision 360** is a production-grade livestock identification and health monitoring kiosk. Traditional livestock tracking relies on ear tags or RFID microchips, which can be lost, tampered with, or expensive. CattleVision 360 uses **bovine rhinoglyphic biometrics**—unique muzzle prints comparable to human fingerprints—to authenticate cattle with high precision.

Combined with an automated **CowGPT RAG Veterinary Advisory Engine**, the system instantly scans uploaded kiosk imagery, verifies bovine features via a strict YOLOv8 confidence gate, extracts $512$-dimensional feature vectors using a custom ArcFace ResNet18 model, queries a FAISS index of over 4,900 cattle records, and generates localized veterinary diagnoses and WhatsApp-ready alerts in Hindi and English.

---

## 🏗️ System Architecture & Inference Pipeline

```text
 [ Kiosk Image Upload / Webcam Capture ]
                   │
                   ▼
       [ YOLOv8n Muzzle Detector ] ──(Confidence < 0.35: Reject 400 Bad Request)──> ❌ Non-Bovine Rejection
                   │
                   ▼ (Bounding Box Crop)
     [ CLAHE Rhinoglyphic Enhancement ]
                   │
                   ▼
     [ PyTorch ResNet18 Embedder (512-d) ]
                   │
                   ▼
        [ FAISS IndexFlatL2 Vector Search ]
                   │
                   ▼
    [ Dynamic Health & CowGPT RAG Engine ]
                   │
                   ▼
      [ FastAPI JSON Response / UI Dashboard ]
```

---

## 📂 Project Directory Structure

```text
CattleVision-360/
├── api/
│   ├── main.py                  # FastAPI server with strict YOLO gate & dynamic vitals
│   ├── requirements.txt         # Pinned python dependencies
│   ├── models/
│   │   ├── best.pt              # YOLOv8n muzzle detection weights
│   │   └── resnet_arcface.pth   # Custom ResNet18 512-d feature extractor weights
│   └── index/
│       ├── cattle_muzzle_faiss.bin # FAISS vector index (IndexFlatL2, dim=512)
│       └── label_map.json       # FAISS row index -> Cattle ID string mapping
├── frontend/
│   └── index.html               # Single-file Tailwind Kiosk dashboard UI
├── .gitignore                   # Excludes venv, heavy binaries, and local configs
└── README.md                    # Project documentation
```

---

## ⚙️ Quick Start & Local Setup

### 1. Clone the Repository
```cmd
git clone https://github.com/Shaivik-ai/CattleVision_360.git
cd CattleVision_360
```

### 2. Set Up Python Virtual Environment
Navigate to the `api` folder and create a virtual environment:
```cmd
cd api
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
Ensure you have the correct NumPy version and CPU/GPU PyTorch packages installed:
```cmd
pip install "numpy<2"
pip install -r requirements.txt
```

### 4. Start the FastAPI Backend Server
```cmd
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```
*The server will start at `http://localhost:8000`. You can verify it by opening `http://localhost:8000/` in your browser.*

---

## 🖥️ Kiosk Scanner UI Usage

1. Open `frontend/index.html` in any modern web browser (Chrome, Edge, Firefox).
2. The top status bar will automatically ping the backend and show **`Online • Production Mode`**.
3. Choose either **Upload File** or **Webcam** mode to capture or select a cattle muzzle image.
4. Click **Run Biometric Scan** to inspect:
   - Identified Livestock ID & Match Confidence (%)
   - FAISS $L_2$ Distance Score & YOLOv8 Bounding Box Verification
   - Health Vitals (Posture analysis, skin lesion surface area %, rumination hours, fever alerts)
   - CowGPT RAG Veterinary Advisory & One-Click Hindi WhatsApp Alerts

---

## 🔒 Key Engineering Features

- **Strict Out-of-Distribution Rejection:** Implements a confidence threshold ($\ge 0.35$) on YOLOv8 muzzle bounding boxes. If a user uploads a passport photo, human face, or random object without a bovine muzzle, the API instantly rejects it with a `400 Bad Request` and a clear error message.
- **CLAHE Rhinoglyphic Enhancement:** Converts images to grayscale and applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to amplify ridge and bead patterns on the cattle muzzle before feature extraction.
- **Dynamic Telemetry & RAG Engine:** Simulates realistic veterinary conditions (e.g., Early Lumpy Skin Disease, FMD risk, or Healthy State) based on feature embedding distances, rather than returning static dummy data.

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---
<div align="center">
  <sub>Built with ❤️ by Shaivik & Team CattleVision 360</sub>
</div>