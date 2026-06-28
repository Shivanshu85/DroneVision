from __future__ import annotations

from pathlib import Path

# Base Paths
DEMO_DIR: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = DEMO_DIR.parent

# Checkpoint and Config Paths
DEFAULT_CHECKPOINT: Path = REPO_ROOT / "runs" / "phase1" / "best.pth"
DEFAULT_CONFIG: Path = REPO_ROOT / "configs" / "phase1.yaml"

# Example & Asset directories
EXAMPLES_DIR: Path = DEMO_DIR / "examples"
ASSETS_DIR: Path = DEMO_DIR / "assets"

# Ensure directories exist
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# UI & Meta details
APPLICATION_TITLE: str = "DroneVision"
APPLICATION_SUBTITLE: str = "Custom Drone Detection Model"
APPLICATION_DESC: str = "Built Completely From Scratch"
MODEL_VERSION: str = "DroneVision Phase 1"
