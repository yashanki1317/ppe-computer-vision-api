import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def process_video(input_path: Path, output_path: Path, weights_path: Path, conf: float):
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    model = YOLO(str(weights_path))
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Could not read video dimensions.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video. Try a different output path.")

    frame_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = model.predict(frame, conf=conf, verbose=False)[0]
            annotated = result.plot()
            writer.write(annotated)
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")
    finally:
        cap.release()
        writer.release()

    print(f"Done. Processed {frame_count} frames.")
    print(f"Annotated video saved to: {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Annotate a video with YOLO PPE detections.")
    parser.add_argument("--input", required=True, help="Path to source MP4/video.")
    parser.add_argument("--output", default="output/annotated_video.mp4")
    parser.add_argument("--weights", default="models/best.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    process_video(Path(args.input), Path(args.output), Path(args.weights), args.conf)


if __name__ == "__main__":
    main()
