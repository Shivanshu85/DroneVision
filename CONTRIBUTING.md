# Contributing to DroneVision

Thank you for your interest in contributing to DroneVision! This document provides instructions for setting up your local environment and submitting contributions.

## Project Guidelines & Scope

DroneVision is a custom single-class drone detection system. Please note the following strict rules:
- **Scope**: The target class is exclusively Drone (Class ID 0). Do not attempt to add multi-class detection or other targets unless explicitly requested in a project roadmap.
- **Architectural Guardrails**: Ultralytics YOLO, pre-packaged toolkits (Detectron2, MMDetection, RT-DETR), and pre-trained weights are strictly prohibited. The model is built and trained completely from scratch in PyTorch.

## Local Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<username>/DroneVision.git
   cd DroneVision
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   Install in editable mode with development dependencies:
   ```bash
   pip install -e .[dev,api]
   ```

4. **Install Pre-commit Hooks**:
   ```bash
   pre-commit install
   ```

## Development Workflow

### Code Quality and Standards
We enforce style checks and static analysis tools on every commit:
- **Linting & Formatting**: Run `ruff check .` and `ruff format .` to scan and fix issues automatically.
- **Type Checking**: Run `mypy --ignore-missing-imports dronevision/` to check type annotations.
- **Testing**: Run all unit tests with `pytest tests/ -v`.

### Making a Pull Request
1. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your edits. Ensure unit tests are added for any new features or bug fixes.
3. Verify all checks pass (linting, tests) locally.
4. Commit your changes and push them to your fork.
5. Create a Pull Request against the `main` branch. Use the provided PR template.

## Security Disclosures
If you find a security vulnerability, please do NOT create a public issue. Instead, report it privately according to the guidelines in [SECURITY.md](file:///c:/Users/tshiv/OneDrive/Documents/Python/DroneVision/SECURITY.md).
