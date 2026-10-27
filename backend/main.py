from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best.pt"

app = FastAPI(
    title="PPE Detection API",
    description="REST API for detecting PPE classes in uploaded images.",
    version="1.0.0",
)

model = None


@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        # Let the API start so /ping and /docs remain usable, but detections require weights.
        model = None
    else:
        model = YOLO(str(MODEL_PATH))


@app.get("/")
def root():
    return {
        "message": "PPE Detection API",
        "docs": "/docs",
        "health": "/ping",
        "image_detection": "/detect/image",
    }


@app.get("/ping")
def ping():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload a valid image file.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    image_array = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="File could not be decoded as an image.")

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model weights not found. Place best.pt at: {MODEL_PATH}",
        )

    try:
        results = model.predict(image, verbose=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    detections = []
    result = results[0]
    names = result.names
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": names[class_id],
                    "confidence": round(confidence, 4),
                    "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                }
            )

    return {
        "filename": file.filename,
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "count": len(detections),
        "detections": detections,
    }
