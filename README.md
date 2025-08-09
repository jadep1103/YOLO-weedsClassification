# YOLO Weeds Classification

Real-time weed detection and classification system using YOLOv8, designed to identify 12 different weed species for precision agriculture applications.

## 🌿 Overview

This project implements a computer vision solution for automated weed detection using YOLO (You Only Look Once) architecture. The system can process images in real-time and accurately classify weeds, making it suitable for integration with agricultural machinery, drones, and monitoring systems.

### Detected Weed Species
- **Waterhemp** - Common in corn and soybean fields
- **MorningGlory** - Climbing vine weed
- **Purslane** - Succulent broadleaf weed
- **SpottedSpurge** - Low-growing annual weed
- **Carpetweed** - Mat-forming summer annual
- **Ragweed** - Major allergen-producing weed
- **Eclipta** - Moisture-loving broadleaf weed
- **PricklySida** - Thorny broadleaf weed
- **PalmerAmaranth** - Highly competitive pigweed species
- **Sicklepod** - Tall annual legume weed
- **Goosegrass** - Clumping grass weed
- **Cutleafgroundcherry** - Solanaceous broadleaf weed

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for real-time processing)

### Installation
```bash
# Clone repository
git clone <repository-url>
cd YOLO-weedsClassification

# Create virtual environment
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup
Organize your dataset in YOLO format:
```
dataset/
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

Update paths in `custom1.yaml` to point to your dataset location.

## 🔧 Training

### Basic Training
```bash
yolo task=detect mode=train epochs=10 data=custom1.yaml model=yolov8m.pt imgsz=640
```

### Extended Training (Recommended)
```bash
yolo task=detect mode=train epochs=150 data=custom1.yaml model=yolov8m.pt imgsz=640 patience=50 batch=16
```

## 🎥 Real-Time Detection

### Live Webcam Detection
```bash
python yolov8.py
```

### Real-Time Inference Options
```python
from ultralytics import YOLO

model = YOLO("train/weights/best.pt")

# Real-time webcam detection
results = model.predict(source="0", show=True, tracker="custom1.yaml")

# Video file processing
results = model.predict(source="video.mp4", show=True, save=True)

# IP camera stream
results = model.predict(source="rtsp://camera_ip", show=True)

# Batch processing for field images
results = model.predict(source="field_images/", save=True, imgsz=640)
```

### Real-Time Performance
- **Processing Speed**: ~30-60 FPS (depending on hardware)
- **Inference Time**: ~15-30ms per frame
- **Memory Usage**: ~2-4GB GPU memory
- **Detection Accuracy**: 99.5% mAP@0.5

## 📈 Model Performance

Based on 150+ training epochs:
- **mAP@0.5**: 99.5%
- **mAP@0.5-0.95**: 97.5%  
- **Precision**: 99.9%
- **Recall**: 100%

The model demonstrates excellent real-time performance with minimal false positives and high detection accuracy across all weed species.

## 📁 Project Structure
```
YOLO-weedsClassification/
├── yolov8.py              # Real-time detection script
├── confusion_matrix.py    # Evaluation tools
├── custom1.yaml           # Dataset configuration
├── requirements.txt       # Dependencies
├── train/
│   └── weights/
│       └── best.pt        # Trained model weights
├── train2/                # Training results & metrics
└── runs/                  # Validation outputs
```

## 🔧 Configuration

### Dataset Configuration (`custom1.yaml`)
```yaml
train: /path/to/train/images
val: /path/to/val/images
nc: 12  # number of classes
names: ['Waterhemp', 'MorningGlory', 'Purslane', 'SpottedSpurge', 
        'Carpetweed', 'Ragweed', 'Eclipta', 'PricklySida', 
        'PalmerAmaranth', 'Sicklepod', 'Goosegrass', 'Cutleafgroundcherry']
```

## 🎯 Applications

- **Precision Agriculture**: Real-time weed mapping for targeted spraying
- **Field Monitoring**: Automated weed density assessment
- **Research**: Agricultural pest management studies
- **Drone Integration**: Aerial weed detection and mapping
- **Smart Farming**: Integration with autonomous tractors and robots

## 📊 Evaluation
```bash
# Generate confusion matrix and performance metrics
python confusion_matrix.py

# Validate model performance
yolo task=detect mode=val data=custom1.yaml model=train/weights/best.pt
```

## 📚 Resources
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Data Augmentation Guide](https://github.com/MinoruHenrique/data_augmentation_yolov7/tree/master)
- Training visualizations available in `train2/` directory