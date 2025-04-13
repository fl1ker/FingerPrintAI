# FingerPrintAI
A deep learning solution for fingerprint segmentation using U-Net with ResNet18.

## Overview
FingerPrintAI trains a U-Net model on 6000 fingerprint images and predicts segmentation masks for new images using PyTorch.

## Features
- Training on 6000 grayscale fingerprint images (96x103).
- Dice and IoU metrics for evaluation.
- Inference script for predicting masks on new images.

## Installation
```bash
pip install torch torchvision pillow numpy matplotlib
