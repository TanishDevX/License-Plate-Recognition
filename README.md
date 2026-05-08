# 🚗 Automatic License Plate Recognition (ALPR)

> End-to-end license plate detection and OCR using **YOLOv11s** + **PaddleOCR**, trained with 2-stage fine-tuning on a Google T4 GPU.

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://python.org)
[![YOLOv11](https://img.shields.io/badge/YOLOv11s-Ultralytics-purple)](https://github.com/ultralytics/ultralytics)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-3.x-red)](https://github.com/PaddlePaddle/PaddleOCR)
[![Colab](https://img.shields.io/badge/Google%20Colab-T4%20GPU-orange?logo=google-colab)](https://colab.research.google.com)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📊 Model Performance

Evaluated on **2,048 validation images** using the trained `best.pt` checkpoint:

| Metric | Score |
|---|---|
| **mAP@50** | **0.9533** |
| **mAP@50-95** | **0.6520** |
| **Precision** | **0.9686** |
| **Recall** | **0.9125** |

**Inference speed:** 0.9ms preprocess · 6.5ms inference · 1.3ms postprocess per image on Tesla T4

> Model: YOLO11s (fused) — 9.4M parameters · 21.3 GFLOPs · 100 layers

---

## 🧠 Architecture & Training

**Detection:** YOLOv11s fine-tuned on a custom license plate dataset (~2,048 validation images, single class)

**2-Stage Fine-Tuning Strategy:**

| Stage | Epochs | Batch | Frozen Layers | Purpose |
|---|---|---|---|---|
| Stage 1 | 6 | 4 | 10 (backbone frozen) | Warm up detection head |
| Stage 2 | 4 | 2 | 0 (full model) | End-to-end fine-tuning |

- Image size: `640×640`, AMP enabled, device: CUDA (T4)
- Stage 2 loads from Stage 1's `last.pt`

**OCR:** PaddleOCR (English, angle classification enabled)  
**Preprocessing before OCR:** Grayscale → 2.5× bicubic upscale → back to BGR

---

## 🔄 Inference Pipeline

```
Input Image → YOLOv11s Detection → Crop Plate (xyxy)
→ Grayscale + 2.5× Upscale → PaddleOCR → Plate Text
```

---

## 📂 Repository Structure

```
License-Plate-Recognition/
├── ALPR.ipynb       # Full pipeline: training, inference, evaluation
├── models/
│   └── best.pt      # Trained YOLOv11s weights
└── README.md
```

---

## 🚀 Roadmap

- [ ] Real-time video / CCTV stream inference
- [ ] Multi-country plate support via PaddleOCR language configs
- [ ] Vehicle type classification (car, truck, bike, bus)
- [ ] Low-light preprocessing enhancement
- [ ] Streamlit / FastAPI demo

---

## 📖 Reference

Inspired by: [*A Deep Learning-Based System for ALPR Using YOLOv12 and PaddleOCR*](https://www.mdpi.com/3400936) — *Applied Sciences, MDPI*

---

## 👨‍💻 Author

**Tanish Sharma** · B.Tech AI & ML · NIT Kurukshetra

---

> ⭐ Star the repo if you found it useful!
