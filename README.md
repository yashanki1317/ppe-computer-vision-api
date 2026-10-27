# PPE Computer Vision API and Video Processing

Assignment 2 project: process videos with OpenCV and expose YOLO PPE image detection through FastAPI. A Streamlit client sends uploaded images to the API and displays the JSON detections.

## Features
- OpenCV frame-by-frame video processing and annotated MP4 output.
- FastAPI health endpoint (`GET /ping`).
- Image inference endpoint (`POST /detect/image`) returning JSON bounding boxes, class names, IDs, and confidence scores.
- Automatic interactive Swagger docs at `/docs`.
- Streamlit client that calls the API rather than loading YOLO directly.

## Requirements
Python 3.10 or newer is recommended. The trained weights are not committed to this repository.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Place the Assignment 1 trained weights at `models/best.pt`. Do not upload the weights or dataset to GitHub unless your mentor explicitly asks.

## 1. Video processing
Put a short MP4 at `input.mp4`, then run:
```bash
python video_processor.py --input input.mp4 --output output/annotated_video.mp4 --weights models/best.pt
```
The script reads frames with `cv2.VideoCapture`, runs YOLO on each frame, and writes annotated frames with `cv2.VideoWriter`.

## 2. Start FastAPI
From the project root, in Terminal window 1:
```bash
uvicorn backend.main:app --reload --port 8000
```
Open http://127.0.0.1:8000/ping and http://127.0.0.1:8000/docs.

## 3. Test the API
In Swagger UI at `/docs`, open `POST /detect/image`, click **Try it out**, choose an image, and execute. Or use cURL:
```bash
curl -X POST "http://127.0.0.1:8000/detect/image" -F "file=@sample.jpg"
```
A successful response contains `count` and `detections`, with `class_name`, `confidence`, and `bbox` as `[x1, y1, x2, y2]`. A non-image upload should return HTTP 400.

## 4. Start Streamlit
In Terminal window 2, from the project root with the same environment activated:
```bash
streamlit run frontend/app.py
```
The app sends images to `http://127.0.0.1:8000` and displays the API results. Keep the backend running while using the frontend.

## Troubleshooting
- **Model not loaded:** verify `models/best.pt` exists and is the trained Assignment 1 checkpoint.
- **Cannot connect:** start the FastAPI server before Streamlit.
- **422 response:** ensure the upload field is named `file` (the provided app and Swagger UI use this name).
- **OpenCV video output issue:** use a short MP4 input and check that the output folder is writable.
- **Missing multipart support:** reinstall dependencies from `requirements.txt`.

## Limitations
The model predicts the classes it was trained on (Gloves, Vest, helmet, person). It does not independently certify workplace safety compliance.
