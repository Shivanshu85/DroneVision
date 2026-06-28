# Changelog

All notable changes to the DroneVision project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-06-28

This release productionizes the DroneVision research codebase for open-source publication, GitHub repositories, Docker deployments, and Hugging Face Spaces.

### Added
- Root-level entrypoint `app.py` for Hugging Face Spaces compatibility.
- Consolidated `DEPLOYMENT.md` detailing local, Docker, and Hugging Face instructions.
- Dedicated multi-stage `Dockerfile` and `docker-compose.yml` configurations running under non-root users.
- Subdivided requirements structure in `requirements/` (`base.txt`, `training.txt`, `demo.txt`, `dev.txt`).
- GitHub workflow automation (`ci.yml` and `release.yml`).
- Git repository hygiene configs (`.gitattributes`, `.editorconfig`, `.pre-commit-config.yaml`).
- Template documents (`CONTRIBUTING.md`, `LICENSE`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, `.env.example`).
- Standardized `VERSION` tracking file.

### Changed
- Configured Ruff and Mypy parameters in `pyproject.toml`.
- Improved `.gitignore` to unignore the production checkpoint `runs/phase1/best.pth` while safely filtering temporary datasets and local environments.
- Redirected root `requirements.txt` to point directly to `requirements/demo.txt`.
