# 🦺 PPE Detection Using YOLOv11

A real-time object detection system that identifies Personal Protective Equipment — **helmets, vests, gloves, and shoes** — in images, videos, and webcam feeds using a fine-tuned YOLOv11 model, with a Streamlit web interface.

The app reports:
- **Total** PPE items detected
- A **per-type breakdown** (e.g. helmet: 2, vest: 1, gloves: 1)

## Quick Start (one-click)

**Mac/Linux:**
```bash
chmod +x start.sh
./start.sh
```

**Windows:**
Double-click `start.bat` (or run it from Command Prompt).

This will:
1. Check for Python
2. Create a virtual environment (`.venv`)
3. Install all required packages
4. Launch the web app at **http://localhost:8501**

The first run downloads the base YOLOv11 model and installs packages, so it can take a few minutes. Later runs start almost instantly.

> **Note:** out of the box, the app uses the general-purpose pretrained `yolo11n.pt` model — it hasn't been trained specifically on PPE, so detections won't be reliable until you fine-tune it (see below).

## Project Structure

```
ppe_detector/
├── app.py               # Streamlit web app (image/video/webcam UI)
├── train.py              # Fine-tuning script
├── predict.py             # CLI inference script (image/video/webcam)
├── requirements.txt       # Python dependencies
├── start.sh / start.bat   # One-click setup + launch
└── dataset/
    ├── data.yaml          # Dataset config
    ├── images/{train,val,test}
    └── labels/{train,val,test}
```

## Step 1: Get the Dataset

Recommended: **"PPE detection yolov11"** by Safety detection on Roboflow Universe — 2,137 images, 4 classes (helmet, vest, gloves, shoes):
`https://universe.roboflow.com/safety-detection-wdyu7/ppe-detection-yolov11-42krk`

1. Open the link, sign in to Roboflow (free)
2. Click **"Download Dataset"**
3. Choose export format **YOLOv11** (or YOLOv8 — same label format)
4. Download the zip to your computer

Alternative datasets if you want more images or different classes (also on Roboflow Universe, search "PPE" or "helmet"):
- "hathelmetvestno-helmetno-vest" — 15.4k images, includes "not wearing" states for compliance detection
- "Personal Protective Equipment" by PPE — helmet/vest/gloves

If you use a dataset with different class names, update `dataset/data.yaml`'s `names:` list to match exactly what the dataset uses — the app reads class names dynamically, so nothing else needs to change.

## Step 2: Place the Dataset Files

Unzip the downloaded dataset. Roboflow exports usually already come split into `train/valid/test` — copy those folders in (rename `valid` → `val`) so you end up with:
```
dataset/images/train/   dataset/labels/train/
dataset/images/val/     dataset/labels/val/
dataset/images/test/    dataset/labels/test/
```
Also copy over the `data.yaml` that came with the download and compare its `names:` list against `dataset/data.yaml` — make sure they match (order matters).

## Step 3: Train (Fine-tune) the Model

```bash
python train.py --data dataset/data.yaml --epochs 100 --model yolo11n.pt
```

- `--model` — starting checkpoint (`yolo11n.pt` = nano/fastest, `yolo11s.pt`/`yolo11m.pt` = larger/more accurate)
- `--epochs` — training iterations over the dataset
- `--device 0` — use GPU 0 if available (omit for CPU/auto)

Trained weights are saved to `runs/train/ppe_yolov11/weights/best.pt`. The Streamlit app automatically detects and uses this file if present.

No GPU? Use Google Colab (free GPU) — upload your `dataset/` folder there, run the same `train.py`, then download `best.pt` back to your machine.

## Step 4: Monitor Training

Ultralytics prints per-epoch loss and mAP. Check `runs/train/ppe_yolov11/` for sample batches with boxes drawn, a `results.png` plot of loss/mAP curves, and a confusion matrix. Watch for:
- Loss steadily decreasing (not stuck or spiking)
- mAP50 rising and plateauing

## Step 5: Evaluate

Training auto-runs validation at the end and prints mAP50/mAP50-95. If accuracy is poor:
- Add more images, especially for classes with fewer examples
- Check labels are correct (mislabeled boxes hurt more than missing ones)
- Train more epochs, or try a larger model

## Step 6: Run Detection

**Via the web app** (recommended): use `start.sh`/`start.bat`, then open http://localhost:8501. Upload an image or video, or use your webcam.

**Via command line:**
```bash
python predict.py --source path/to/image.jpg --weights runs/train/ppe_yolov11/weights/best.pt
python predict.py --source path/to/video.mp4 --weights runs/train/ppe_yolov11/weights/best.pt
python predict.py --source 0 --weights runs/train/ppe_yolov11/weights/best.pt   # webcam
```

## Manual Setup (if you skip the start scripts)

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Troubleshooting

- **"No module named ultralytics"** — the venv isn't activated, or `pip install -r requirements.txt` failed. Re-run the start script.
- **Webcam doesn't open in `predict.py`** — try a different `--source` index (0, 1, 2...) or check camera permissions.
- **Training is slow** — training on CPU is much slower than GPU. Reduce `--epochs` or `--imgsz`, or use a cloud GPU (e.g. Google Colab) for training, then copy `best.pt` back locally.
- **Detections are inaccurate** — this is expected with the base pretrained model. Fine-tune on the PPE dataset above (Steps 1–3).

## Applications

- Construction site safety compliance monitoring
- Factory/warehouse PPE auditing
- Automated safety alerts from surveillance footage
- Safety training and demonstration tools

## Tech Stack

Python · YOLOv11 (Ultralytics) · PyTorch · OpenCV · Streamlit · NumPy
