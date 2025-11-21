from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import uvicorn
import numpy as np
import cv2

app = FastAPI()

# Allow Flutter to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model
model = YOLO("best.pt")   # replace with your model path

# POST endpoint for detection
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

# ✅ GET endpoint for testing connectivity
@app.get("/")
def home():
    return {"message": "YOLO Detection API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
