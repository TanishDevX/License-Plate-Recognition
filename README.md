# 🚗 Automatic License Plate Recognition (ALPR)

> An end-to-end license plate detection and recognition pipeline using **YOLOv11** for plate localization and **PaddleOCR** for text extraction — trained with 2-stage fine-tuning on a Google T4 GPU.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-purple?logo=yolo)](https://github.com/ultralytics/ultralytics)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-3.x-red?logo=paddlepaddle)](https://github.com/PaddlePaddle/PaddleOCR)
[![Colab](https://img.shields.io/badge/Run%20on-Google%20Colab-orange?logo=google-colab)](https://colab.research.google.com)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📌 Overview

This project implements a full **ALPR pipeline** combining state-of-the-art object detection and OCR:

1. **YOLOv11** detects and localizes the license plate bounding box
2. The cropped plate is preprocessed (grayscale + 2.5× bicubic upscaling)
3. **PaddleOCR** reads the text with angle correction enabled

Trained using a **2-stage fine-tuning strategy** on Google Colab's T4 GPU — first freezing backbone layers to stabilize early training, then unfreezing all layers for full fine-tuning.

---

## ✨ Key Features

| Feature | Details |
|---|---|
| 🎯 Plate Detection | YOLOv11 trained on custom dataset (YOLOv11 format, `data.yaml`) |
| ✂️ Smart Cropping | Bounding box crop from YOLO `xyxy` output |
| 🔍 Preprocessing | Grayscale conversion + 2.5× bicubic upscaling before OCR |
| 🔤 OCR Engine | PaddleOCR with English language + angle classification |
| 🏋️ 2-Stage Training | Stage 1: 6 epochs, frozen backbone; Stage 2: 4 epochs, full fine-tune |
| 📊 Evaluation | mAP50, mAP50-95, Precision, Recall via `model.val()` |

---

## 🛠️ Tech Stack

- **Detection:** YOLOv11 (Ultralytics)
- **OCR:** PaddleOCR (PaddlePaddle 3.2.2)
- **Image Processing:** OpenCV
- **Visualization:** Matplotlib
- **Environment:** Google Colab (T4 GPU)
- **Storage:** Google Drive (dataset + model weights)

---

## 📂 Repository Structure

```
License-Plate-Recognition/
│
├── ALPR.ipynb          # Full training + inference pipeline
├── models/
│   └── best.pt         # Trained YOLOv11 weights
└── README.md
```

---

## ▶️ Getting Started

### Option 1 — Google Colab *(Recommended)*

1. Open `ALPR.ipynb` in Colab → set runtime to **GPU (T4)**
2. Mount Google Drive and place your dataset at:
   ```
   /content/drive/MyDrive/Dataset/
   ```
   Ensure the dataset includes a `data.yaml` file in YOLOv11 format.
3. Run all cells sequentially — training, inference, and evaluation are all included.

### Option 2 — Run Locally

```bash
pip install ultralytics paddlepaddle==3.2.2 paddleocr opencv-python matplotlib
apt-get install -y libgl1-mesa-glx   # Linux only
```

Then run the notebook in Jupyter or VS Code with a CUDA-enabled GPU.

---

## 🔄 Pipeline

```
Input Image
    │
    ▼
YOLOv11 Detection  →  Bounding Box (xyxy)
    │
    ▼
Crop License Plate Region
    │
    ▼
Grayscale + 2.5× Bicubic Upscaling
    │
    ▼
PaddleOCR (angle correction enabled)
    │
    ▼
Extracted Plate Text
```

---

## 🏋️ Training Strategy

The model uses a **2-stage fine-tuning approach** to maximize accuracy with limited compute:

| Stage | Epochs | Batch Size | Frozen Layers | Purpose |
|---|---|---|---|---|
| Stage 1 | 6 | 4 | 10 (backbone) | Warm up detection head |
| Stage 2 | 4 | 2 | 0 (full model) | End-to-end fine-tuning |

- Image size: `640×640`
- Mixed precision training (AMP enabled)
- Stage 2 initializes from Stage 1's `last.pt`

---

## 📊 Evaluation

After training, the model is evaluated using:

```python
results = model.val()
# Outputs: mAP50, mAP50-95, Precision, Recall
```

---

## 🚀 Roadmap

- [ ] Real-time video stream inference (webcam / CCTV)
- [ ] Multi-country / multilingual plate support via PaddleOCR language configs
- [ ] Vehicle type classification (car, truck, bike, bus)
- [ ] Low-light image enhancement preprocessing
- [ ] Streamlit or FastAPI web demo
- [ ] Extended training with more epochs on higher-VRAM GPU

---

## 📖 References

- **Research Inspiration:** [*A Deep Learning-Based System for Automatic License Plate Recognition Using YOLOv12 and PaddleOCR*](https://www.mdpi.com/3400936) — *Applied Sciences, MDPI*
- [Ultralytics YOLOv11 Docs](https://docs.ultralytics.com)
- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [OpenCV Documentation](https://docs.opencv.org)

---

## 👨‍💻 Author

**Tanish Sharma**  
B.Tech — Artificial Intelligence & Machine Learning  
NIT Kurukshetra

---

## 📜 License

This project is licensed under the [MIT License](LICENSE) — free to use, modify, and distribute with attribution.

---

> ⭐ If this project helped you or sparked ideas, consider giving it a star!
