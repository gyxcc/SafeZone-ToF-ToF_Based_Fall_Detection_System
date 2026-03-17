"""
ToF-Based Fall Detection System
================================

Real-time fall detection for elderly care using:
- 3D ToF depth camera (100x100)
- YOLOv8-pose skeleton extraction
- Random Forest temporal classification
- State machine delayed alerting

Main entry point: python tof_fall_detection.py --port COM3 --preview
"""

__version__ = "1.0.0"
__author__ = "Your Name"
