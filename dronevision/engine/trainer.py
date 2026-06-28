"""
DroneTrainer — Complete training loop for DroneVision.

Features:
    - CUDA auto-detection with CUDA → MPS → CPU fallback.
    - Mixed-precision training via torch.amp.GradScaler.
    - Gradient clipping to prevent NaN loss from exploding gradients.
    - Warmup + cosine annealing LR schedule.
    - Per-epoch evaluation with mAP50 tracking.
    - Best-model and last-model checkpoint saving.
    - MLflow experiment tracking (local, optional).
    - Training resumption from checkpoint.
    - EarlyStopping callback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# mlflow is optional — if not installed, use a no-op stub so that the
# training loop runs without experiment tracking.
try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

    class _MlflowStub:
        """No-op stub matching the mlflow API surface used by DroneTrainer."""

        def set_tracking_uri(self, *a, **kw) -> None: ...
        def set_experiment(self, *a, **kw) -> None: ...
        def start_run(self): return self
        def __enter__(self): return self
        def __exit__(self, *a): ...
        def log_params(self, *a, **kw) -> None: ...
        def log_metric(self, *a, **kw) -> None: ...

    mlflow = _MlflowStub()  # type: ignore[assignment]

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dronevision.data.collate import drone_collate_fn
from dronevision.data.dataset import DroneDataset
from dronevision.engine.callbacks import EarlyStopping, WarmupCosineScheduler
from dronevision.engine.evaluator import DroneEvaluator
from dronevision.inference.nms import non_max_suppression
from dronevision.loss.detection_loss import DroneDetectionLoss
from dronevision.models.detector import DroneDetector
from dronevision.utils.logger import configure_logging, get_logger
from dronevision.utils.reproducibility import get_device, set_seed

logger = get_logger(__name__)


class DroneTrainer:
    """
    Manages the full training lifecycle for DroneDetector.

    Args:
        cfg:         Full configuration dict from load_config().
        resume_path: Optional path to a checkpoint to resume training from.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        resume_path: str | Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.train_cfg = cfg["train"]
        self.model_cfg = cfg["model"]
        self.data_cfg = cfg["data"]
        self.log_cfg = cfg["logging"]

        self.resume_path = Path(resume_path) if resume_path else None

        # Set up logging
        configure_logging()

        # Reproducibility
        set_seed(self.train_cfg["seed"])

        # Device
        self.device = get_device(prefer_cuda=True)

        # Build dataloaders
        self.train_loader, self.val_loader = self._build_dataloaders()

        # Build model
        self.model = self._build_model()

        # Loss function
        self.criterion = DroneDetectionLoss(cfg)

        # Optimizer
        self.optimizer = self._build_optimizer()

        # Scheduler
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=self.train_cfg.get("warmup_epochs", 3),
            total_epochs=self.train_cfg["epochs"],
        )

        # Mixed precision scaler (only on CUDA)
        use_amp = self.train_cfg.get("mixed_precision", False) and self.device.type == "cuda"
        self.scaler: torch.amp.GradScaler | None = (
            torch.amp.GradScaler("cuda") if use_amp else None
        )
        if use_amp:
            logger.info("Mixed precision (AMP) enabled")
        else:
            logger.info("Mixed precision disabled (device=%s)", self.device.type)

        # Evaluator
        self.evaluator = DroneEvaluator(num_classes=self.model_cfg["num_classes"])

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.train_cfg.get("patience", 15),
            mode="max",
        )

        # State
        self.start_epoch: int = 0
        self.best_map50: float = 0.0
        self.checkpoint_dir = Path(self.train_cfg["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Resume
        if self.resume_path is not None:
            self._resume_from_checkpoint()

        logger.info("DroneTrainer ready | epochs=%d | device=%s", self.train_cfg["epochs"], self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> dict[str, float]:
        """
        Run the complete training loop.

        Returns:
            Dict with final evaluation metrics.
        """
        mlflow.set_tracking_uri(self.log_cfg["mlflow_tracking_uri"])
        mlflow.set_experiment(self.log_cfg["experiment_name"])

        with mlflow.start_run():
            # Log hyperparameters
            self._log_hyperparams()

            for epoch in range(self.start_epoch, self.train_cfg["epochs"]):
                logger.info(
                    "Epoch %d/%d — LR=%.6f",
                    epoch + 1,
                    self.train_cfg["epochs"],
                    self.optimizer.param_groups[0]["lr"],
                )

                # Training epoch
                train_metrics = self._train_epoch(epoch)

                # Update LR
                self.scheduler.step()

                # Validation
                val_results = self._validate()

                # Log to MLflow
                self._log_metrics(epoch, train_metrics, val_results)

                # Save checkpoints
                is_best = val_results.map50 > self.best_map50
                if is_best:
                    self.best_map50 = val_results.map50
                    self._save_checkpoint(epoch, val_results.to_dict(), name="best")
                    logger.info("New best mAP50=%.4f — checkpoint saved", self.best_map50)

                if (epoch + 1) % self.train_cfg.get("save_period", 5) == 0:
                    self._save_checkpoint(epoch, val_results.to_dict(), name="last")

                # Early stopping
                if self.early_stopping(val_results.map50):
                    logger.info("Early stopping triggered at epoch %d", epoch + 1)
                    break

            # Always save last checkpoint
            self._save_checkpoint(epoch, val_results.to_dict(), name="last")
            mlflow.log_metric("best_mAP50", self.best_map50)

        logger.info("Training complete | best mAP50=%.4f", self.best_map50)
        return val_results.to_dict()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch. Returns averaged loss metrics."""
        self.model.train()
        totals: dict[str, float] = {
            "total_loss": 0.0,
            "box_loss": 0.0,
            "obj_loss": 0.0,
            "cls_loss": 0.0,
        }
        n_batches = len(self.train_loader)
        log_interval = self.log_cfg.get("log_interval", 10)

        pbar = tqdm(
            self.train_loader,
            desc=f"Train [{epoch+1}/{self.train_cfg['epochs']}]",
            leave=False,
        )

        for batch_idx, (images, targets, _) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Forward pass (with AMP if enabled)
            if self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    raw_preds = self.model(images)
                    loss, metrics = self.criterion(
                        raw_preds, targets, self.model.anchors
                    )
                self.scaler.scale(loss).backward()
                # Gradient clipping before unscale+step
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_cfg.get("gradient_clip", 10.0),
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                raw_preds = self.model(images)
                loss, metrics = self.criterion(
                    raw_preds, targets, self.model.anchors
                )
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_cfg.get("gradient_clip", 10.0),
                )
                self.optimizer.step()

            for k, v in metrics.items():
                totals[k] += v

            if (batch_idx + 1) % log_interval == 0:
                pbar.set_postfix({
                    "loss": f"{metrics['total_loss']:.4f}",
                    "box":  f"{metrics['box_loss']:.4f}",
                    "obj":  f"{metrics['obj_loss']:.4f}",
                })

        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    @torch.no_grad()
    def _validate(self):
        """Run validation and return EvalResults."""
        self.model.eval()
        self.evaluator.reset()

        conf_thresh = self.cfg["inference"]["conf_threshold"]
        iou_thresh = self.cfg["inference"]["iou_threshold"]

        for images, targets, _ in tqdm(self.val_loader, desc="Val", leave=False):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Decoded predictions: (B, N_total, 5+nc)
            decoded = self.model(images)

            # NMS per image
            B = images.shape[0]
            preds_list = []
            for b in range(B):
                dets = non_max_suppression(
                    decoded[b:b+1],
                    conf_threshold=conf_thresh,
                    iou_threshold=iou_thresh,
                )
                preds_list.append(dets[0])  # may be None

            self.evaluator.update(preds_list, targets.cpu())

        return self.evaluator.compute()

    def _build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        aug_cfg = self.cfg.get("augmentation", {})

        train_ds = DroneDataset(
            image_dir=self.data_cfg["train"],
            label_dir=str(Path(self.data_cfg.get("label_dir", "datasets/labels")) / "train"),
            img_size=self.model_cfg["image_size"],
            augment=aug_cfg.get("enabled", True),
            aug_cfg=aug_cfg,
            is_training=True,
        )
        val_ds = DroneDataset(
            image_dir=self.data_cfg["val"],
            label_dir=str(Path(self.data_cfg.get("label_dir", "datasets/labels")) / "val"),
            img_size=self.model_cfg["image_size"],
            augment=False,
            is_training=False,
        )

        nw = self.data_cfg.get("num_workers", 4)
        pin = self.data_cfg.get("pin_memory", True) and self.device.type == "cuda"

        train_loader = DataLoader(
            train_ds,
            batch_size=self.train_cfg["batch_size"],
            shuffle=True,
            num_workers=nw,
            pin_memory=pin,
            collate_fn=drone_collate_fn,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.train_cfg["batch_size"],
            shuffle=False,
            num_workers=nw,
            pin_memory=pin,
            collate_fn=drone_collate_fn,
        )

        logger.info(
            "DataLoaders: train=%d images | val=%d images | batch=%d | workers=%d",
            len(train_ds), len(val_ds),
            self.train_cfg["batch_size"], nw,
        )
        return train_loader, val_loader

    def _build_model(self) -> DroneDetector:
        model = DroneDetector(self.cfg)
        model.to(self.device)
        return model

    def _build_optimizer(self) -> torch.optim.Optimizer:
        opt_name = self.train_cfg.get("optimizer", "adamw").lower()
        lr = self.train_cfg["lr"]
        wd = self.train_cfg.get("weight_decay", 0.0005)

        if opt_name == "adamw":
            opt = torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=wd
            )
        elif opt_name == "sgd":
            mom = self.train_cfg.get("momentum", 0.937)
            opt = torch.optim.SGD(
                self.model.parameters(), lr=lr,
                momentum=mom, weight_decay=wd, nesterov=True,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        logger.info("Optimizer: %s | lr=%.4f | wd=%.4f", opt_name, lr, wd)
        return opt

    def _resume_from_checkpoint(self) -> None:
        if not self.resume_path.exists():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {self.resume_path}"
            )
        checkpoint = torch.load(self.resume_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.start_epoch = checkpoint.get("epoch", 0) + 1
        self.best_map50 = checkpoint.get("metrics", {}).get("mAP50", 0.0)
        logger.info(
            "Resumed from %s at epoch %d (best mAP50=%.4f)",
            self.resume_path, self.start_epoch, self.best_map50,
        )

    def _save_checkpoint(
        self,
        epoch: int,
        metrics: dict[str, float],
        name: str = "last",
    ) -> None:
        path = self.checkpoint_dir / f"{name}.pth"
        opt_state = self.scaler.state_dict() if self.scaler else None
        self.model.save_checkpoint(
            path,
            epoch=epoch,
            metrics=metrics,
            optimizer_state=self.optimizer.state_dict(),
            scaler_state=opt_state,
        )

    def _log_hyperparams(self) -> None:
        params = {
            "image_size": self.model_cfg["image_size"],
            "batch_size": self.train_cfg["batch_size"],
            "epochs": self.train_cfg["epochs"],
            "optimizer": self.train_cfg.get("optimizer", "adamw"),
            "lr": self.train_cfg["lr"],
            "weight_decay": self.train_cfg.get("weight_decay", 0.0005),
            "warmup_epochs": self.train_cfg.get("warmup_epochs", 3),
            "mixed_precision": self.train_cfg.get("mixed_precision", False),
            "lambda_box": self.cfg["loss"]["lambda_box"],
            "lambda_obj": self.cfg["loss"]["lambda_obj"],
            "lambda_cls": self.cfg["loss"]["lambda_cls"],
            "anchor_threshold": self.cfg["loss"]["anchor_threshold"],
            "num_classes": self.model_cfg["num_classes"],
        }
        mlflow.log_params(params)

    def _log_metrics(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_results,
    ) -> None:
        step = epoch + 1
        mlflow.log_metric("train/total_loss", train_metrics["total_loss"], step=step)
        mlflow.log_metric("train/box_loss", train_metrics["box_loss"], step=step)
        mlflow.log_metric("train/obj_loss", train_metrics["obj_loss"], step=step)
        mlflow.log_metric("train/cls_loss", train_metrics["cls_loss"], step=step)
        mlflow.log_metric("val/mAP50", val_results.map50, step=step)
        mlflow.log_metric("val/mAP50-95", val_results.map50_95, step=step)
        mlflow.log_metric("val/precision", val_results.precision, step=step)
        mlflow.log_metric("val/recall", val_results.recall, step=step)
        mlflow.log_metric("val/f1", val_results.f1, step=step)
        mlflow.log_metric("lr", self.optimizer.param_groups[0]["lr"], step=step)
        logger.info(
            "[Epoch %d] Loss=%.4f | %s",
            step, train_metrics["total_loss"], val_results,
        )
