"""Engine subpackage — lazy imports to avoid hard dependency on mlflow at collection time."""

from __future__ import annotations


def __getattr__(name: str):
    if name == "DroneTrainer":
        from dronevision.engine.trainer import DroneTrainer
        return DroneTrainer
    if name == "DroneEvaluator":
        from dronevision.engine.evaluator import DroneEvaluator
        return DroneEvaluator
    raise AttributeError(f"module 'dronevision.engine' has no attribute {name!r}")


__all__ = ["DroneTrainer", "DroneEvaluator"]
