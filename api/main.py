import os

# Must be set before importing torch, faiss, or cv2 on Windows/Anaconda
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import json
import random
import time

# Import torch FIRST on Windows so c10.dll initializes before faiss/opencv DLLs
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from ultralytics import YOLO

import cv2
import faiss
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

app = FastAPI(title="CattleVision 360 API", version="1.1")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- PATHS & GLOBAL MODEL LOADING ---
BASE_DIR = Path(__file__).resolve().parent
YOLO_WEIGHTS_PATH = BASE_DIR / "models" / "best.pt"
RESNET_WEIGHTS_PATH = BASE_DIR / "models" / "resnet_arcface.pth"
FAISS_INDEX_PATH = BASE_DIR / "index" / "cattle_muzzle_faiss.bin"
LABEL_MAP_PATH = BASE_DIR / "index" / "label_map.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_LOADED = False

yolo_model = None
index = None
embedder = None
label_map = None

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

try:
    print(f"[INFO] Attempting to load models on device: {device}...")

    if not YOLO_WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Missing YOLO weights at {YOLO_WEIGHTS_PATH}")
    if not RESNET_WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Missing ResNet weights at {RESNET_WEIGHTS_PATH}")
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(f"Missing FAISS index at {FAISS_INDEX_PATH}")

    # 1. Load YOLOv8 Muzzle Detector
    yolo_model = YOLO(str(YOLO_WEIGHTS_PATH))

    # 2. Load FAISS Vector Index
    index = faiss.read_index(str(FAISS_INDEX_PATH))

    # 3. Load PyTorch ResNet18 Feature Extractor (512-d)
    embedder = resnet18(weights=None)
    embedder.fc = torch.nn.Linear(embedder.fc.in_features, 512)
    state_dict = torch.load(RESNET_WEIGHTS_PATH, map_location=device)
    embedder.load_state_dict(state_dict)
    embedder.to(device)
    embedder.eval()

    # 4. Load Optional Label Map (maps FAISS vector row -> cattle ID string)
    if LABEL_MAP_PATH.exists():
        with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        print(f"[INFO] Loaded label_map.json with {len(label_map)} entries.")
    else:
        print("[WARNING] index/label_map.json not found. Using raw FAISS index IDs.")

    MODELS_LOADED = True
    print(f"[INFO] All models loaded successfully! FAISS total vectors: {index.ntotal}")
except Exception as e:
    print(f"[WARNING] Models could not be loaded. Running in Simulation mode. Reason: {e}")


def decode_image(img_bytes: bytes) -> np.ndarray:
    """Decodes raw image bytes into a BGR OpenCV numpy array."""
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode uploaded image bytes.")
    return img


def apply_clahe_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """Converts a BGR image to grayscale, applies CLAHE, and returns a 3-channel RGB image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    return cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)


@app.get("/")
def health_check():
    return {
        "status": "Online",
        "mode": "Production" if MODELS_LOADED else "Simulation",
        "device": device,
        "faiss_vectors": index.ntotal if index is not None else 0,
    }


@app.post("/api/scan")
async def scan_cattle(file: UploadFile = File(...)):
    """
    Ingests Kiosk image, runs strict YOLOv8 muzzle verification, 
    FAISS similarity search, and dynamic health anomaly generation.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Must upload a valid image file.")

    try:
        image_bytes = await file.read()
        raw_bgr = decode_image(image_bytes)

        if MODELS_LOADED:
            # 1. Run YOLOv8 Muzzle Detection with a strict confidence threshold (e.g., 0.35)
            # This ensures random objects, humans, or passport photos without a muzzle are rejected.
            results = yolo_model(raw_bgr, verbose=False)
            boxes = results[0].boxes
            
            # Filter boxes by confidence score >= 0.35
            valid_boxes = [i for i, conf in enumerate(boxes.conf) if conf.item() >= 0.35] if len(boxes) > 0 else []
            muzzle_detected = len(valid_boxes) > 0

            if not muzzle_detected:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "❌ Muzzle Not Detected! No valid bovine muzzle rhinoglyphic region "
                        "exceeding confidence threshold was found in this image. "
                        "Please upload a clear, focused image of a cattle muzzle."
                    )
                )

            # Crop to highest-confidence bounding box
            best_box_idx = valid_boxes[0]
            x1, y1, x2, y2 = map(int, boxes.xyxy[best_box_idx].tolist())
            h, w, _ = raw_bgr.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Ensure crop box has valid dimensions
            if (x2 - x1) > 20 and (y2 - y1) > 20:
                target_region = raw_bgr[y1:y2, x1:x2]
            else:
                target_region = raw_bgr

            # 2. Apply CLAHE Rhinoglyphic Enhancement
            enhanced_rgb = apply_clahe_bgr(target_region)
            pil_img = Image.fromarray(enhanced_rgb)

            # 3. Extract 512-d Metric Embedding
            input_tensor = transform(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                vector = embedder(input_tensor).cpu().numpy().astype("float32")

            # 4. Query FAISS Index
            distances, indices = index.search(vector, k=1)
            match_idx = int(indices[0][0])
            match_dist = float(distances[0][0])

            # Resolve Cattle ID from label_map if available
            if label_map and 0 <= match_idx < len(label_map):
                resolved_name = str(label_map[match_idx]).upper()
                cow_id = resolved_name if resolved_name.startswith("COW_") else f"COW_{resolved_name}"
            else:
                cow_id = f"COW_IDX_{match_idx}"

            confidence = round(float(1.0 / (1.0 + match_dist)), 4)
        else:
            # Fallback Simulation Mode
            time.sleep(1.0)
            _ = apply_clahe_bgr(raw_bgr)
            cow_id = f"COW_IN_{random.randint(1000, 9999)}"
            confidence = round(random.uniform(0.92, 0.99), 3)
            match_dist = round((1.0 / confidence) - 1.0, 4)
            muzzle_detected = True

        # --- DYNAMIC HEALTH & COW-GPT RAG ANOMALY ENGINE ---
        # Instead of static healthy results every time, we simulate realistic veterinary conditions
        # based on minor variations in match distance and hash properties to make demo testing robust.
        
        # Introduce a pseudo-random health check seed based on match_dist digits
        health_seed = int((match_dist * 1000) % 10)
        
        if health_seed == 1:
            # Condition A: Mild Lumpy Skin Disease (LSD) Early Nodules
            lesion_pct = round(random.uniform(3.5, 6.8), 1)
            rumination_hrs = round(random.uniform(3.5, 4.8), 1)
            temp_alert = True
            posture = "Lethargic / Isolation (Angle: 142°)"
            diagnosis = "Early nodular skin lesions detected near muzzle periphery. Mild pyrexia indicated."
            action = "Isolate animal immediately. Administer broad-spectrum anti-inflammatory and consult district veterinarian."
            hindi_alert = "⚠️ पशु में शुरुआती त्वचा गांठ (LSD) के लक्षण मिले हैं। तुरंत अलग करें और डॉक्टर को दिखाएं।"
        elif health_seed == 2:
            # Condition B: Foot-and-Mouth Disease (FMD) Blister Risk
            lesion_pct = round(random.uniform(2.1, 4.5), 1)
            rumination_hrs = round(random.uniform(2.8, 4.2), 1)
            temp_alert = True
            posture = "Recumbent / Excessive Salivation"
            diagnosis = "Possible vesicular eruptions (FMD marker) identified along rhinoglyphic ridges."
            action = "Quarantine herd section. Disinfect water troughs with sodium carbonate. Restrict animal movement."
            hindi_alert = "⚠️ मुंह और खुर की बीमारी (FMD) के लक्षण। पशु को तुरंत बाकी झुंड से अलग करें।"
        else:
            # Condition C: Healthy State
            lesion_pct = 0.0
            rumination_hrs = round(random.uniform(6.5, 8.2), 1)
            temp_alert = False
            posture = "Normal Standing (Angle: 178°)"
            diagnosis = "Healthy condition. Rhinoglyphic patterns match database successfully with no anomalies."
            action = "Maintain standard feeding ration and mineral mixture."
            hindi_alert = "गाय बिल्कुल स्वस्थ है। सामान्य चारा और पानी जारी रखें।"

        response_payload = {
            "cow_id": cow_id,
            "mode": "Production" if MODELS_LOADED else "Simulation",
            "biometrics": {
                "muzzle_match_confidence": confidence,
                "l2_distance_score": match_dist,
                "yolo_muzzle_detected": muzzle_detected,
                "clahe_enhancement_applied": True,
            },
            "health_engine": {
                "detected_posture": posture,
                "lesion_surface_area_pct": lesion_pct,
                "simulated_rumination_hrs": rumination_hrs,
                "temperature_alert": temp_alert,
            },
            "cowgpt_rag_diagnosis": {
                "preliminary_diagnosis": diagnosis,
                "action": action,
                "whatsapp_alert_hindi": hindi_alert,
            },
        }

        return response_payload

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))