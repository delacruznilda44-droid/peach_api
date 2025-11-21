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

# Make a writable cache directory
cache_dir = "/tmp/ultralytics"
os.makedirs(cache_dir, exist_ok=True)

model_path = os.path.join(os.getcwd(), "best.pt")  # points to current working directory
model = YOLO(model_path)

@app.get("/")
def home():
    return {"message": "YOLO API is running"}

@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    # Read uploaded image
    contents = await image.read()
    np_image = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    # Run YOLO prediction
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
