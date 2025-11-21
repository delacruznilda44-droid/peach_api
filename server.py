from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2
import os

app = FastAPI()

# Allow all origins (for Flutter app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to your YOLO model
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "best.pt")

# Load YOLO model
model = YOLO(MODEL_PATH)

@app.get("/")
def home():
    return {"message": "YOLO API is running"}

# Main detection endpoint
@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    contents = await image.read()
    np_image = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    results = model(img)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls = int(box.cls)
        conf = float(box.conf)
        label = model.names[cls]

        detections.append({
            "class": label,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2]
        })

    return {"detections": detections}

# Alias endpoint for existing APKs using /predict
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    return await detect(image)
