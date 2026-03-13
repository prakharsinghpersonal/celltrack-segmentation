# CellTrack — Microscopy Instance Segmentation

Automated instance segmentation pipeline for noisy, low-contrast microscopy images using Mask R-CNN (PyTorch).

## Overview

CellTrack performs end-to-end instance segmentation on microscopy data:
- **Mask R-CNN** with ResNet-50 FPN backbone (pretrained on ImageNet, fine-tuned for microscopy)
- **Domain-specific augmentation** — contrast jittering, elastic deformation, Gaussian noise injection
- **Automated IoU evaluation** — computes mAP at standard COCO IoU thresholds [0.5:0.95]
- Achieves **mAP 0.82** on test set

## Architecture

```
Raw Microscopy Images
        │
        ▼
┌──────────────────────────┐
│  Preprocessing           │  CLAHE, median filter, normalization
│  (OpenCV)                │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Data Augmentation       │  Random crop, contrast jitter, elastic deform
│  (albumentations)        │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Mask R-CNN              │
│  ├─ ResNet-50 Backbone   │  Feature extraction
│  ├─ FPN                  │  Multi-scale feature maps
│  ├─ RPN                  │  Region proposals
│  └─ Mask Head            │  Pixel-level instance masks
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Evaluation              │  mAP, IoU, precision, recall
│  (COCO-style)            │
└──────────────────────────┘
```

## Key Results

| Metric | Value |
|:---|:---|
| mAP @ IoU [0.5:0.95] | **0.82** |
| mAP @ IoU 0.5 | 0.91 |
| Inference speed | ~120ms per image (GPU) |
| Training data | ~3,000 annotated microscopy images |

## Project Structure

```
CellTrack/
├── README.md
├── requirements.txt
├── config.py                  # Hyperparameters and paths
├── dataset.py                 # Custom PyTorch Dataset + augmentation
├── model.py                   # Mask R-CNN setup and fine-tuning
├── train.py                   # Training loop with LR scheduling
├── evaluate.py                # IoU/mAP evaluation engine
├── preprocess.py              # CLAHE, denoising, normalization
├── utils.py                   # Visualization, logging helpers
└── augmentations.py           # Microscopy-specific augmentation pipeline
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Train
python train.py --data_dir ./data --epochs 50 --batch_size 4

# Evaluate
python evaluate.py --model_path ./checkpoints/best_model.pth --data_dir ./data/test

# Preprocess raw images
python preprocess.py --input_dir ./raw_images --output_dir ./processed
```

## Technical Details

- **Preprocessing**: CLAHE (Contrast-Limited Adaptive Histogram Equalization) for local contrast enhancement, median filtering for noise reduction
- **Loss**: Multi-task loss (classification + bounding box regression + mask prediction)
- **Optimizer**: SGD with momentum (0.9), weight decay (1e-4), cosine annealing LR schedule
- **Augmentation**: albumentations for simultaneous image + mask + bbox transforms

## Technologies

Python, PyTorch, torchvision, OpenCV, albumentations, NumPy, Matplotlib, scikit-image
