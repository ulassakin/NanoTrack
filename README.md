# NanoTrack: Real-time Small Object Detection & Tracking (v0.1 - Image Inference)

Real-time small object detection and tracking on edge devices using EdgeYOLO, Norfair and TensorRT.

This repository contains a fine-tuned EdgeYOLO model for small object detection, starting with **image inference**.  
The project is designed for edge devices (e.g., Jetson Orin NX) and will evolve step by step into a full real-time detection + tracking system.

## Current Features
- Fine-tuned model (VisDrone dataset)
- Inference on static images with bounding box visualization

## Usage
```bash
python inference_image.py --engine best.engine --meta best.json --source test.jpg
