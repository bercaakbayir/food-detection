# Food & Vessel Detection Pipeline

A computer vision pipeline for detecting vessels (cups, bowls, etc.), identifying their content, and calculating fullness and physical dimensions.

## Features
- **Vessel Detection**: YOLOv10 for robust object detection.
- **Surface Segmentation**: YOLOv8-seg for food and drink surface masks.
- **Liquid Level Detection**: Hybrid approach using depth estimation and edge gradients for transparent liquids.
- **Size Estimation**: Automatic distance calculation based on standard vessel dimensions.
- **Modular Structure**: Clean architecture for easy extensibility.

## Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
python main.py --path data/glass.png
```

## Project Structure
- `src/`: Core logic modules.
- `models/`: YOLO model weights.
- `data/`: Input images.
- `main.py`: CLI entry point.
