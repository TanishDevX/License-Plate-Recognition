<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=200&section=header&text=ALPR&fontSize=80&fontColor=ffffff&fontAlignY=35&desc=Automatic%20License%20Plate%20Recognition&descAlignY=58&descSize=22&descColor=a78bfa" width="100%"/>

<br/>

[![mAP50](https://img.shields.io/badge/mAP%4050-95.33%25-6d28d9?style=for-the-badge&logoColor=white)](.)
[![Precision](https://img.shields.io/badge/Precision-96.86%25-7c3aed?style=for-the-badge)](.)
[![Recall](https://img.shields.io/badge/Recall-91.25%25-8b5cf6?style=for-the-badge)](.)
[![Inference](https://img.shields.io/badge/Inference-6.5ms%20%2F%20image-a78bfa?style=for-the-badge)](.)

<br/>

[![Python](https://img.shields.io/badge/Python_3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![YOLOv11](https://img.shields.io/badge/YOLOv11s-Ultralytics-111827?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-3.x-0062B1?style=flat-square)](https://github.com/PaddlePaddle/PaddleOCR)
[![CUDA](https://img.shields.io/badge/CUDA-Tesla_T4-76b900?style=flat-square&logo=nvidia&logoColor=white)](.)
[![License](https://img.shields.io/badge/License-MIT-gray?style=flat-square)](LICENSE)

<br/>

*A production-grade ALPR system combining **YOLOv11s** plate detection with **PaddleOCR** text extraction,  
trained via 2-stage fine-tuning on a Tesla T4 GPU — achieving **95.33% mAP@50** on 2,048 validation images.*

</div>

---

## 📊 Results

<div align="center">

| Metric | Score | Context |
|:---|:---:|:---|
| **mAP@50** | **95.33%** | Primary detection accuracy benchmark |
| **mAP@50-95** | **65.20%** | Strict IoU-averaged detection quality |
| **Precision** | **96.86%** | Low false positive rate |
| **Recall** | **91.25%** | Strong plate detection coverage |
| **Inference** | **6.5ms / image** | On Tesla T4 GPU |
| **Val Set** | **2,048 images** | 2,195 instances, single class |

</div>

> **Model:** YOLO11s (fused) · 9.41M parameters · 21.3 GFLOPs · 100 layers · PyTorch 2.9 + CUDA 12.6

---

## 🔄 Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Input Image                                               │
│       │                                                     │
│       ▼                                                     │
│   YOLOv11s Detection ──► Bounding Box (xyxy coords)        │
│       │                                                     │
│       ▼                                                     │
│   Crop Plate Region                                         │
│       │                                                     │
│       ▼                                                     │
│   Grayscale → 2.5× Bicubic Upscale → BGR Conversion        │
│       │                                                     │
│       ▼                                                     │
│   PaddleOCR  (English · Angle Classification ON)           │
│       │                                                     │
│       ▼                                                     │
│   Extracted Plate Text ✅                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🏋️ Training Strategy

A **2-stage progressive fine-tuning** approach was used to maximise accuracy under T4 GPU memory constraints:

**Stage 1 — Warm Up** `6 epochs · batch 4 · backbone frozen (freeze=10)`  
Trains only the detection head while keeping YOLOv11s's pretrained backbone locked. Prevents early-epoch gradient chaos and lets the head specialise on license plates fast.

**Stage 2 — Full Fine-Tune** `4 epochs · batch 2 · all layers unfrozen (freeze=0)`  
Loads Stage 1's `last.pt` and unfreezes the entire network for end-to-end optimisation. Smaller batch compensates for the increased gradient computation.

```
Pretrained YOLOv11s
        │
   [Stage 1] ── 6 epochs, frozen backbone, imgsz=640, AMP ──► stage1/last.pt
        │
   [Stage 2] ── 4 epochs, full model, imgsz=640, AMP       ──► best.pt ✅
```

---

## 🛠️ Tech Stack

| Component | Technology |
|:---|:---|
| Detection Model | YOLOv11s (Ultralytics 8.3.230) |
| OCR Engine | PaddleOCR + PaddlePaddle 3.2.2 |
| Image Processing | OpenCV |
| Training Hardware | Tesla T4 · 15GB VRAM |
| Framework | PyTorch 2.9 + CUDA 12.6 |
| Environment | Google Colab |

---

## 📂 Repository Structure

```
License-Plate-Recognition/
├── ALPR.ipynb        # End-to-end pipeline: training → inference → evaluation
├── models/
│   └── best.pt       # Trained YOLOv11s weights
└── README.md
```

---

## 🚀 Roadmap

- [ ] Real-time video / CCTV stream inference
- [ ] Multi-country & multilingual plate support
- [ ] Vehicle type classification (car, truck, bike, bus)
- [ ] Low-light and motion-blur preprocessing
- [ ] FastAPI + Streamlit live demo deployment
- [ ] Extended training on higher-VRAM GPU for mAP@50-95 improvement

---

## 📖 Reference

> Inspired by: [*A Deep Learning-Based System for Automatic License Plate Recognition Using YOLOv12 and PaddleOCR*](https://www.mdpi.com/3400936) — *Applied Sciences, MDPI*

---

<div align="center">

**Tanish Sharma** · B.Tech AI & ML · NIT Kurukshetra

<br/>

*If this project helped you or you found it interesting, consider leaving a ⭐*

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=100&section=footer" width="100%"/>

</div>
