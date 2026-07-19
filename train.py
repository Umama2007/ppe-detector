"""
train.py - Fine-tune YOLOv11 on a custom PPE (Personal Protective Equipment) dataset.

Usage:
    python train.py --data dataset/data.yaml --epochs 100 --model yolo11n.pt

Before running:
    1. Download a PPE dataset (e.g. "PPE detection yolov11" by Safety detection on
       Roboflow Universe: https://universe.roboflow.com/safety-detection-wdyu7/ppe-detection-yolov11-42krk)
       in YOLOv11 format, or annotate your own images.
    2. Put images/labels into dataset/images/{train,val,test} and
       dataset/labels/{train,val,test} in YOLO format.
    3. Edit dataset/data.yaml if your class names/paths differ.
"""

import argparse
import os
from pathlib import Path
import torch

from ultralytics import YOLO


def check_dataset(data_yaml: str) -> bool:
    """Basic sanity check that images/labels actually exist before training."""
    root = Path(data_yaml).parent
    train_dir = root / "images" / "train"
    val_dir = root / "images" / "val"

    if not train_dir.exists() or not any(train_dir.iterdir()):
        print(f"⚠️  No training images found in {train_dir}")
        print("   Add annotated images before training. See README.md.")
        return False

    if not val_dir.exists() or not any(val_dir.iterdir()):
        print(f"⚠️  No validation images found in {val_dir}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv11 on PPE dataset")
    parser.add_argument("--data", type=str, default="dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Pretrained base model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default 10 for quick CPU run)")
    parser.add_argument("--imgsz", type=int, default=320, help="Training image size (smaller for faster CPU)")
    parser.add_argument("--batch", type=int, default=4, help="Batch size (lower for CPU memory limits)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use: 'cpu' or CUDA index like '0'")
    parser.add_argument("--name", type=str, default="ppe_yolov11", help="Run name")
    args = parser.parse_args()

    if not check_dataset(args.data):
        print("\nTraining aborted: dataset is incomplete.")
        return

    print(f"Loading pretrained model: {args.model}")
    model = YOLO(args.model)

    print("Starting fine-tuning...")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        name=args.name,
        project="runs/train",
    )

    print("Running validation on best checkpoint...")
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.4f}  mAP50-95: {metrics.box.map:.4f}")

    best_weights = Path("runs/train") / args.name / "weights" / "best.pt"
    print(f"\nDone. Best weights saved at: {best_weights}")
    print("Copy or point to this file in predict.py / app.py to use your fine-tuned model.")


if __name__ == "__main__":
    main()
