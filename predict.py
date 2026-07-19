"""
predict.py - Run PPE (helmet/vest/gloves/shoes) detection on an image, a video file,
or a live webcam feed.

Usage:
    python predict.py --source path/to/image.jpg
    python predict.py --source path/to/video.mp4
    python predict.py --source 0                 # webcam
    python predict.py --source path/to/image.jpg --weights runs/train/ppe_yolov11/weights/best.pt
"""

import argparse
import time
from collections import Counter

import cv2
from ultralytics import YOLO


def type_breakdown(model, boxes) -> Counter:
    """Count detections per PPE item type (class name)."""
    counts = Counter()
    for box in boxes:
        cls = int(box.cls[0])
        counts[model.names[cls]] += 1
    return counts


def draw_summary(frame, counts: Counter):
    total = sum(counts.values())
    lines = [f"Total PPE Items: {total}"] + [f"{name}: {n}" for name, n in counts.items()]
    box_h = 26 * len(lines) + 14
    cv2.rectangle(frame, (0, 0), (300, box_h), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        color = (0, 255, 0) if i == 0 else (255, 255, 255)
        cv2.putText(
            frame, line, (10, 27 + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )
    return frame


def run_on_image(model, source, conf):
    results = model.predict(source=source, conf=conf, save=True)
    r = results[0]
    counts = type_breakdown(model, r.boxes)
    total = sum(counts.values())

    print(f"Total PPE items detected: {total}")
    if total == 0:
        print("  (none found)")
    for name, n in counts.items():
        print(f"  - {name}: {n}")

    for box in r.boxes:
        cls = int(box.cls[0])
        confidence = float(box.conf[0])
        print(f"    detection -> {model.names[cls]} (confidence {confidence:.2f})")

    print(f"Annotated image saved under: {r.save_dir}")


def run_on_stream(model, source, conf):
    """Handles both video files and webcam (source=0, 1, ...)."""
    cap_source = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(cap_source)

    if not cap.isOpened():
        print(f"Could not open source: {source}")
        return

    window_name = "PPE Detection - press 'q' to quit"
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=conf, verbose=False)
        r = results[0]
        annotated = r.plot()
        counts = type_breakdown(model, r.boxes)
        annotated = draw_summary(annotated, counts)

        now = time.time()
        fps = 1 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(
            annotated, f"FPS: {fps:.1f}", (10, annotated.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
        )

        cv2.imshow(window_name, annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Run PPE detection")
    parser.add_argument("--source", type=str, required=True,
                         help="Path to image/video, or webcam index (e.g. 0)")
    parser.add_argument("--weights", type=str, default="yolo11n.pt",
                         help="Path to trained weights (defaults to base pretrained model)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use: 'cpu' or CUDA index like '0'")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    model = YOLO(args.weights, device=args.device)

    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    if str(args.source).lower().endswith(image_exts):
        run_on_image(model, args.source, args.conf)
    else:
        run_on_stream(model, args.source, args.conf)


if __name__ == "__main__":
    main()
