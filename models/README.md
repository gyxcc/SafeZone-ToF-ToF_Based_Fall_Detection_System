# Model Weights

This folder should contain the pre-trained model weights.

## Required Files

| File | Description | Size |
|------|-------------|------|
| `best.pt` | Fine-tuned YOLOv8s-pose for depth images | ~22 MB |
| `fall_classifier_v6.joblib` | Random Forest temporal classifier | ~5 MB |

## Download Links

> ⚠️ Model weights are not included in the repository due to size limitations.

Download from:
- **Google Drive**: [Link to be added]
- **GitHub Releases**: Check the Releases page of this repository

## After Download

1. Place `best.pt` in this folder (`models/`)
2. Place `fall_classifier_v6.joblib` in this folder (`models/`)
3. Update paths in `src/config.py` if needed

## Training Your Own Models

See `src/train_rf_classifier_v6.py` for training the Random Forest classifier.

For YOLOv8-pose fine-tuning, refer to [Ultralytics documentation](https://docs.ultralytics.com/tasks/pose/).
