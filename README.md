# ToF-Based Fall Detection System

A real-time elderly fall detection system using 3D Time-of-Flight (ToF) depth camera, YOLOv8-pose skeleton extraction, and Random Forest temporal classification.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Features

- **Privacy-Preserving**: Uses depth data instead of RGB cameras
- **Real-Time Processing**: 30 FPS on NVIDIA RTX 4060 GPU
- **High Accuracy**: 97% classification accuracy with 100% fall recall
- **Smart Alerting**: State machine with 10-second confirmation to reduce false alarms
- **Multi-Class Detection**: Distinguishes forward falls, backward falls, and normal activities

## System Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│  ToF Camera │───►│ YOLOv8-pose  │───►│ Random Forest   │───►│ Fall Monitor │
│  (100x100)  │    │ (5 keypoints)│    │ (50-frame window)│    │ (State Machine)│
└─────────────┘    └──────────────┘    └─────────────────┘    └──────────────┘
     10 FPS           ~30 FPS              3-class             Delayed Alert
```

### Processing Pipeline

1. **Depth Acquisition**: ToF sensor captures 100×100 depth images at 10 FPS
2. **Pose Estimation**: Fine-tuned YOLOv8s-pose extracts 5 upper-body keypoints (nose, shoulders, hips)
3. **Temporal Classification**: Random Forest analyzes 50-frame (5-second) sliding window
4. **Smart Alerting**: State machine monitors falls and only triggers alert if no recovery within 10 seconds

## Project Structure

```
tof_fall_detection_release/
├── src/
│   ├── tof_fall_detection.py    # Main entry point
│   ├── config.py                # Configuration settings
│   ├── depth_converter.py       # ToF raw data to depth image
│   ├── pose_estimator.py        # YOLOv8-pose wrapper
│   ├── fall_classifier_v6.py    # Random Forest classifier
│   ├── fall_monitor.py          # State machine for delayed alerting
│   ├── keypoint_filter.py       # Keypoint validation
│   ├── pipeline_realtime.py     # Real-time processing pipeline
│   ├── visualizer.py            # Visualization utilities
│   ├── train_rf_classifier_v6.py# Training script
│   └── utils.py                 # Helper functions
├── configs/
│   ├── fall_detection.yaml      # YOLO training config
│   └── fall_pose.yaml           # Pose estimation config
├── models/                      # Model weights (download required)
│   ├── best.pt                  # Fine-tuned YOLOv8s-pose
│   └── fall_classifier_v6.joblib# Trained RF classifier
├── docs/                        # Documentation
├── environment.yml              # Conda environment
└── README.md
```

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/tof-fall-detection.git
cd tof-fall-detection
```

### 2. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate tfgpu
```

### 3. Download Model Weights

Download pre-trained weights and place in `models/` folder:

| Model | Size | Download |
|-------|------|----------|
| YOLOv8s-pose (fine-tuned) | ~22 MB | [Google Drive](#) |
| RF Classifier v6 | ~5 MB | [Google Drive](#) |

### 4. Update Config Paths

Edit `src/config.py` to point to your model locations:

```python
YOLO_MODEL_PATH = "models/best.pt"
RF_MODEL_PATH = "models/fall_classifier_v6.joblib"
```

## Usage

### Real-Time Detection (ToF Camera)

```bash
cd src
python tof_fall_detection.py --port COM3 --preview
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--port` | COM3 | Serial port for ToF camera |
| `--preview` | False | Show real-time preview window |
| `--alert-delay` | 10.0 | Seconds before triggering alert |
| `--no-monitor` | False | Disable delayed alerting |
| `--threshold` | 0.60 | Fall confidence threshold |
| `--window` | 50 | Detection window size (frames) |

### Example Commands

```bash
# Basic usage with preview
python tof_fall_detection.py --port COM3 --preview

# Faster alert (5 seconds)
python tof_fall_detection.py --port COM3 --preview --alert-delay 5

# Immediate alerting (no monitoring delay)
python tof_fall_detection.py --port COM3 --preview --no-monitor

# Record depth frames for later analysis
python tof_fall_detection.py --port COM3 --preview --record
```

## Fall Monitor State Machine

The system uses a 3-state finite state machine to reduce false alarms:

```
┌─────────┐    Fall detected    ┌────────────┐
│  IDLE   │ ──────────────────► │ MONITORING │
│ (Green) │                     │ (Orange)   │
└─────────┘                     └────────────┘
     ▲                                │
     │   Person recovers         10s timeout
     │   (stands up)                  │
     │                                ▼
     │                          ┌─────────┐
     └──────────────────────────│ ALERTED │
            Manual reset        │ (Red)   │
                                └─────────┘
```

| State | Description | Display |
|-------|-------------|---------|
| IDLE | Normal monitoring | Green skeleton |
| MONITORING | Fall detected, waiting for recovery | Orange "WATCHING" + countdown |
| ALERTED | Confirmed fall, alert triggered | Red "ALERT!" |

## Hardware Requirements

### ToF Camera
- **Model**: 100×100 depth sensor
- **Interface**: Serial (RS-232/USB)
- **Baud Rate**: 1,000,000
- **Frame Rate**: 10 FPS

### Computing
- **GPU**: NVIDIA GPU with CUDA support (tested on RTX 4060)
- **RAM**: 8GB+ recommended
- **Python**: 3.8+

## Training Your Own Model

### 1. Collect Training Data

```bash
python tof_fall_detection.py --port COM3 --record --record-dir training_data/my_falls
```

### 2. Annotate with YOLO

Run pose estimation on recorded frames to generate keypoints.csv.

### 3. Train RF Classifier

```bash
python train_rf_classifier_v6.py
```

## Performance

| Metric | Value |
|--------|-------|
| Classification Accuracy | 97% |
| Fall Recall | 100% |
| Backward Fall Recall | 100% |
| Inference FPS | ~30 (GPU) |
| Detection Latency | <500ms |
| False Alarm Reduction | ~85% (with monitor) |

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{tof_fall_detection_2026,
  author = {Your Name},
  title = {ToF-Based Fall Detection System},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/YOUR_USERNAME/tof-fall-detection}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for pose estimation
- [scikit-learn](https://scikit-learn.org/) for Random Forest implementation
