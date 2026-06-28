# DroneVision Deployment & Pipeline Guide

This document describes how to deploy the DroneVision application locally, using Docker, or to Hugging Face Spaces, and outlines the automated CI/CD pipeline.

---

## 1. Local Deployment

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (optional, fallbacks to CPU automatically)
- Git & Git LFS

### Installation
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/Shivanshu85/DroneVision.git
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
1. Build the image:
   ```bash
   docker build -t dronevision-demo:1.0.0 .
   ```
2. Run the container:
   ```bash
   docker run -p 7860:7860 dronevision-demo:1.0.0
   ```

---

## 3. Hugging Face Spaces Deployment

The repository is structured to enable fully automated deployments to Hugging Face Spaces. The deployment script isolates only the necessary runtime files, keeping the Space clean and lightweight.

### Space Layout Isolation
When pushed to Hugging Face, only the following runtime structure is transferred:
```
Hugging Face Space Root/
├── app.py                     # Root entrypoint wrapper
├── requirements.txt           # Flat dependencies list
├── VERSION                    # Version number file
├── .gitattributes             # Git LFS config for weights
├── README.md                  # Spaces title/emoji metadata block
├── dronevision/               # Core library code
├── demo/                      # Gradio UI components
├── configs/                   # Runtime YAML settings
└── runs/
    └── phase1/
        └── best.pth           # Trained model weights (LFS tracked)
```
All training scripts, tests, notebooks, local environments, and temporary files are excluded from the Hugging Face Space repository to optimize build speed and startup times.

---

## 4. Automated CI/CD Workflows (GitHub Actions)

DroneVision uses four automated workflows under `.github/workflows/` to ensure continuous quality control:

1. **Continuous Integration (`ci.yml`)**:
   - Runs on every push and pull request.
   - Executes code checks using Ruff (linting and format reviews).
   - Validates package import paths (`import dronevision`).
   - Validates dependency packages structure via `pip check`.
   - Runs the full unit test suite via PyTest.
   - Fails immediately if any step encounters an error.

2. **Docker Validation (`docker_validation.yml`)**:
   - Runs on every push and pull request.
   - Builds the Docker image from the local `Dockerfile`.
   - Starts the container in background mode.
   - Verifies successful container startup by polling `http://localhost:7860/` for a successful HTTP 200 response code.
   - Cleans up the test container immediately.

3. **Hugging Face Deployment (`huggingface_deployment.yml`)**:
   - Runs automatically on pushes to the `main` branch.
   - Collects and isolates only the required runtime files in a temporary space layout.
   - Syncs the LFS weight checkpoints securely.
   - Forces a git push to the Hugging Face Spaces remote repository, triggering an automatic rebuild on Hugging Face.

4. **Release Automation (`release.yml`)**:
   - Triggers when a new version tag is pushed (e.g., `v1.0.0`).
   - Validates the codebase.
   - Builds the source and binary wheel distributions.
   - Automatically creates a GitHub Release and attaches the builds as assets.

---

## 5. Security & Secrets Management

To enable automated deployments to Hugging Face Spaces, you must configure a Hugging Face Access Token in your GitHub repository:

### Required GitHub Secrets
1. **`HF_TOKEN`**: A Hugging Face write-access token.
   - Generate one at: [Hugging Face Settings > Tokens](https://huggingface.co/settings/tokens).
   - Configure it in your GitHub repo: **Settings > Secrets and variables > Actions > New repository secret**.
   - Set the name to `HF_TOKEN` and paste the token as the value.

---

## 6. Troubleshooting

### Logging warnings on Windows
If you run the application on Windows, you might notice encoding warnings:
```
UnicodeEncodeError: 'charmap' codec can't encode character '\u2192'
```
This is harmless and occurs because the Windows command line console defaults to `cp1252` encoding instead of `utf-8`. To resolve this, run your shell with:
```bash
set PYTHONIOENCODING=utf-8
```

### Docker build failure
If Docker fails to load the weights, verify that you have Git LFS installed locally and the checkpoint file `runs/phase1/best.pth` has its full size (~37.7 MB) rather than just the LFS metadata placeholder.
