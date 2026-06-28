"""
DroneVision — Custom single-class drone detection and counting system.

Architecture:
    Image → Backbone → Neck → Detection Head → Post-processing → Drone Count

Classes:
    0 = Drone (UAV)

Phase 1 scope:
    - Dataset validation and conversion
    - Custom backbone, neck, detection head
    - Training and evaluation pipelines
    - Image inference and drone counting

Usage:
    from dronevision.models.detector import DroneDetector
    from dronevision.inference.predictor import DronePredictor
"""

__version__ = "0.1.0"
__author__ = "DroneVision Team"
