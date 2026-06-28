# DroneVision Deployment Guide

This document describes how to deploy the DroneVision Gradio application locally, using Docker, or to Hugging Face Spaces.

---

## 1. Local Deployment

### Prerequisites
- Python 3.11+
- CUDA-capable GPU (optional, but recommended for speed. Falling back to CPU is fully supported)
- Git & Git LFS

### Installation
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/<username>/DroneVision.git
   cd DroneVision
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. Install the package in editable mode with demo and API dependencies:
   ```bash
   pip install -e .[dev,api]
   ```

### Running the Application
Launch the Gradio demo application using the root entrypoint:
```bash
python app.py
```
By default, the application will bind to `http://127.0.0.1:7860`. You can configure host and port settings using environment variables or a `.env` file (see `.env.example`).

---

## 2. Docker Deployment

We provide a production-ready, multi-stage Docker configuration that builds a minimal image and runs as a non-root user.

### Run with Docker Compose (Recommended)
To build and spin up the Gradio application container:
```bash
docker-compose up --build
```
The application will be accessible at `http://localhost:7860`.

### Manual Docker Build and Run
If you prefer to run the Docker command directly:
1. Build the image:
   ```bash
   docker build -t dronevision-demo:1.0.0 .
   ```
2. Run the container:
   ```bash
   docker run -p 7860:7860 dronevision-demo:1.0.0
   ```

### Configuration & Security
- **Non-root Execution**: The container runs under the `appuser` user inside the `appgroup` group.
- **Port Mapping**: Container port `7860` is exposed and can be mapped to any host port.
- **Checkpoints**: The container copies `runs/phase1/best.pth` at build time to enable immediate out-of-the-box predictions.

---

## 3. Hugging Face Spaces Deployment

The repository includes a root-level `app.py` and structured dependencies to support seamless deployment to Hugging Face Spaces.

### Step-by-Step Deployment
1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces).
   - Select **Gradio** as the SDK.
   - Choose the appropriate hardware (CPU basic is free; GPU space will run inference faster).
2. Set up Git LFS on your machine if not already installed:
   ```bash
   git lfs install
   ```
3. Add the Hugging Face Space repository as a git remote:
   ```bash
   git remote add hf https://huggingface.co/spaces/<your-username>/<your-space-name>
   ```
4. Push the code and tracking configurations to Hugging Face:
   ```bash
   git push hf main
   ```

### Space Customization (Metadata)
If you deploy this repository as a Hugging Face Space, you can optionally prepend the following YAML metadata block at the very top of the repository's `README.md` to customize the space configuration:
```yaml
---
title: DroneVision
emoji: 🚁
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.19.0
app_file: app.py
pinned: false
---
```
This metadata is ignored by local runs but processed by the Hugging Face deployment pipeline to configure the Gradio environment and SDK version.
