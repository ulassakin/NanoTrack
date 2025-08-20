# NanoTrack: Real-time Small Object Detection & Tracking (v0.1 - Image Inference)

Real-time small object detection and tracking on edge devices using EdgeYOLO, Norfair and TensorRT.

This repository contains a fine-tuned EdgeYOLO model for small object detection using VisDrone dataset, starting with **image inference**.  
The project is designed for edge devices (e.g., Jetson Orin NX) and will evolve step by step into a full real-time detection + tracking system with a user interface.

## Current Features
- Fine-tuned EdgeYOLO model(VisDrone dataset)
- Object detection and classification on static images with bounding box visualization

## Usage
```bash
python detect.py --weights edgeyolo_visdrone.pth --source detect_video_EdgeYOLO.mp4  --conf-thres 0.3 --nms-thres 0.5 --input-size 640 640 --fp16 --batch 1 --save-dir ./output 
