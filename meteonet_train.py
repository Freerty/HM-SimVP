"""HM-SimVP MeteoNet training. Dependencies: torch, numpy, h5py, tqdm."""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Iterable
import h5py
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from model import HMSimVP, MODEL_NAME

def resolve_project_path(path):
    return Path(path).expanduser().resolve()

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class MeteoNetH5Dataset(Dataset):
    """Lazy per-worker HDF5 reader with MeteoNet's existing normalization."""

    def __init__(
        self,
        h5_path: str | os.PathLike[str],
        group: str,
        input_length: int = 12,
        output_length: int = 6,
        raw_clip_max: float = 833.0,
        augment: bool = False,
    ) -> None:
        self.h5_path = str(resolve_project_path(h5_path))
        self.group = str(group)
        self.input_length = int(input_length)
        self.output_length = int(output_length)
        self.raw_clip_max = float(raw_clip_max)
        self.augment = bool(augment)
        self._file: Optional[h5py.File] = None
        self._pid: Optional[int] = None

        if not os.path.isfile(self.h5_path):
            raise FileNotFoundError(f"MeteoNet data file not found: {self.h5_path}")
        with h5py.File(self.h5_path, "r") as handle:
            if self.group not in handle or "images" not in handle[self.group]:
                raise KeyError(f"Expected HDF5 dataset '{self.group}/images'.")
            shape = tuple(handle[self.group]["images"].shape)
        if len(shape) != 4 or shape[1] < self.input_length + self.output_length:
            raise ValueError(
                f"Expected [N,T,H,W] with T >= {self.input_length + self.output_length}, "
                f"got {shape}."
            )
        self.shape = shape
        self.log_scale = float(math.log1p(self.raw_clip_max))

    def __len__(self) -> int:
        return int(self.shape[0])

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_file"] = None
        state["_pid"] = None
        return state

    def _images(self):
        pid = os.getpid()
        if self._file is None or self._pid != pid:
            if self._file is not None:
                self._file.close()
            self._file = h5py.File(self.h5_path, "r")
            self._pid = pid
        return self._file[self.group]["images"]

    def _normalize(self, array: np.ndarray) -> np.ndarray:
        # Raw MeteoNet values are 0.01 mm accumulated over five minutes.
        array = np.clip(array.astype(np.float32, copy=False), 0.0, None)
        return np.log1p(array) / self.log_scale

    @staticmethod
    def _augment_pair(
        history: np.ndarray,
        future: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        # This matches the geometric/intensity augmentation used by the
        # preceding MeteoNet training file, but is never called for validation.
        if np.random.rand() < 0.5:
            history = np.flip(history, axis=2)
            future = np.flip(future, axis=2)
        if np.random.rand() < 0.5:
            history = np.flip(history, axis=1)
            future = np.flip(future, axis=1)
        rotation = int(np.random.randint(0, 4))
        if rotation:
            history = np.rot90(history, rotation, axes=(1, 2))
            future = np.rot90(future, rotation, axes=(1, 2))
        if np.random.rand() < 0.5:
            scale = float(np.random.uniform(0.8, 1.2))
            history = np.clip(history * scale, 0.0, 1.0)
            future = np.clip(future * scale, 0.0, 1.0)
        if np.random.rand() < 0.3:
            noise_std = float(np.random.uniform(0.01, 0.05))
            noise = np.random.normal(0.0, noise_std, history.shape).astype(np.float32)
            history = np.clip(history + noise, 0.0, 1.0)
        return np.ascontiguousarray(history), np.ascontiguousarray(future)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = np.asarray(self._images()[index], dtype=np.float32)
        history = self._normalize(sequence[: self.input_length])
        future = self._normalize(
            sequence[self.input_length : self.input_length + self.output_length]
        )
        if self.augment:
            history, future = self._augment_pair(history, future)
        return torch.from_numpy(history), torch.from_numpy(future)

def build_datasets(config: Dict[str, Any]):
    common = dict(
        h5_path=config["data_file"],
        input_length=config["input_length"],
        output_length=config["output_length"],
        raw_clip_max=config["raw_clip_max"],
    )
    # Separate objects are deliberate: validation must not inherit training
    # augmentation merely because both subsets refer to the HDF5 train group.
    train_source = MeteoNetH5Dataset(
        group="train",
        augment=bool(config["train_augmentation"]),
        **common,
    )
    val_source = MeteoNetH5Dataset(group="train", augment=False, **common)
    test_dataset = MeteoNetH5Dataset(group="test", augment=False, **common)

    total = len(train_source)
    val_size = int(float(config["validation_ratio"]) * total)
    if val_size <= 0 or val_size >= total:
        raise ValueError(
            f"validation_ratio={config['validation_ratio']} gives invalid size {val_size}/{total}."
        )
    train_size = total - val_size
    train_dataset = Subset(train_source, range(0, train_size))
    val_dataset = Subset(val_source, range(train_size, total))
    split = {
        "train": len(train_dataset),
        "validation": len(val_dataset),
        "test": len(test_dataset),
        "validation_start_index": train_size,
    }
    return train_dataset, val_dataset, test_dataset, split

def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    config: Dict[str, Any],
    generator: Optional[torch.Generator] = None,
) -> DataLoader:
    workers = int(config["num_workers"])
    kwargs: Dict[str, Any] = dict(
        dataset=dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=bool(config["pin_memory"]),
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    if workers > 0:
        kwargs["persistent_workers"] = bool(config["persistent_workers"])
        kwargs["prefetch_factor"] = int(config["prefetch_factor"])
    return DataLoader(**kwargs)

def split_model_output(output: Any) -> tuple[torch.Tensor, Any]:
    if isinstance(output, dict):
        prediction = output.get("prediction")
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("Model output dict must contain tensor key 'prediction'.")
        return prediction, output.get("aux_loss")
    if isinstance(output, (tuple, list)):
        return output[0], output[1] if len(output) > 1 else None
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Unsupported model output type: {type(output)!r}")
    return output, None

class MCRPLoss(nn.Module):
    """The established MCRP training objective, without a spatial mask."""

    def __init__(
        self,
        patch_size: int = 8,
        base_weight: float = 0.10,
        mean_weight: float = 0.10,
        contrast_weight: float = 0.05,
        log_contrast_limit: float = 0.45,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.base_weight = float(base_weight)
        self.mean_weight = float(mean_weight)
        self.contrast_weight = float(contrast_weight)
        self.log_contrast_limit = float(log_contrast_limit)
        self.last_components: Dict[str, float] = {}

    def _patch_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(
            tensor,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            ceil_mode=True,
            count_include_pad=False,
        )

    def _patch_peak(self, tensor: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(
            tensor,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            ceil_mode=True,
        )

    def forward(self, prediction, target, aux_loss=None):
        if not isinstance(aux_loss, dict):
            raise ValueError("MCRP model output must include its auxiliary dictionary.")
        base = aux_loss.get("base_prediction")
        patch_shift = aux_loss.get("patch_shift")
        patch_log_contrast = aux_loss.get("patch_log_contrast")
        if not all(isinstance(x, torch.Tensor) for x in (base, patch_shift, patch_log_contrast)):
            raise TypeError("MCRP auxiliary output is missing required tensors.")

        target_mean = self._patch_mean(target)
        base_detached = base.detach()
        base_mean = self._patch_mean(base_detached)
        target_shift = target_mean - base_mean
        target_peak = self._patch_peak(target)
        base_peak = self._patch_peak(base_detached)
        ratio = (target_peak - target_mean).clamp_min(1e-4) / (
            (base_peak - base_mean).clamp_min(1e-4)
        )
        target_log_contrast = ratio.log().clamp(
            -self.log_contrast_limit,
            self.log_contrast_limit,
        )
        final_mse = F.mse_loss(prediction, target)
        base_mse = F.mse_loss(base, target)
        mean_huber = F.smooth_l1_loss(patch_shift, target_shift)
        contrast_huber = F.smooth_l1_loss(patch_log_contrast, target_log_contrast)
        total = (
            final_mse
            + self.base_weight * base_mse
            + self.mean_weight * mean_huber
            + self.contrast_weight * contrast_huber
        )
        self.last_components = {
            "final_mse": float(final_mse.detach().cpu()),
            "base_mse": float(base_mse.detach().cpu()),
            "mean_huber": float(mean_huber.detach().cpu()),
            "contrast_huber": float(contrast_huber.detach().cpu()),
            "total": float(total.detach().cpu()),
        }
        return total

class MetricAccumulator:
    def __init__(self, output_length: int, thresholds: Iterable[float], raw_clip_max: float):
        self.output_length = int(output_length)
        self.thresholds = tuple(float(x) for x in thresholds)
        self.log_scale = float(math.log1p(raw_clip_max))
        self.sse = 0.0
        self.sae = 0.0
        self.count = 0
        self.pred_min = float("inf")
        self.pred_max = -float("inf")
        self.true_min = float("inf")
        self.true_max = -float("inf")
        self.counts = self._new_counts()
        self.lead_sse = [0.0] * self.output_length
        self.lead_sae = [0.0] * self.output_length
        self.lead_count = [0] * self.output_length
        self.lead_counts = [self._new_counts() for _ in range(self.output_length)]

    def _new_counts(self):
        return {threshold: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for threshold in self.thresholds}

    def _to_mmh(self, normalized: torch.Tensor) -> torch.Tensor:
        # raw unit: 0.01 mm / 5 min, hence raw * 0.12 = mm/h.
        return torch.expm1(normalized.float() * self.log_scale) * 0.12

    @staticmethod
    def _update_binary(table, prediction, target, thresholds):
        for threshold in thresholds:
            pred_rain = prediction >= threshold
            true_rain = target >= threshold
            table[threshold]["tp"] += int((pred_rain & true_rain).sum().item())
            table[threshold]["fp"] += int((pred_rain & ~true_rain).sum().item())
            table[threshold]["fn"] += int((~pred_rain & true_rain).sum().item())
            table[threshold]["tn"] += int((~pred_rain & ~true_rain).sum().item())

    @torch.no_grad()
    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = self._to_mmh(prediction.detach()).cpu()
        target = self._to_mmh(target.detach()).cpu()
        error = prediction - target
        self.sse += float(error.square().sum().item())
        self.sae += float(error.abs().sum().item())
        self.count += error.numel()
        self.pred_min = min(self.pred_min, float(prediction.min().item()))
        self.pred_max = max(self.pred_max, float(prediction.max().item()))
        self.true_min = min(self.true_min, float(target.min().item()))
        self.true_max = max(self.true_max, float(target.max().item()))
        self._update_binary(self.counts, prediction, target, self.thresholds)
        for lead in range(self.output_length):
            lead_error = error[:, lead]
            self.lead_sse[lead] += float(lead_error.square().sum().item())
            self.lead_sae[lead] += float(lead_error.abs().sum().item())
            self.lead_count[lead] += lead_error.numel()
            self._update_binary(
                self.lead_counts[lead],
                prediction[:, lead],
                target[:, lead],
                self.thresholds,
            )

    @staticmethod
    def _finish_counts(table):
        result = {}
        eps = 1e-8
        for threshold, values in table.items():
            tp, fp, fn, tn = (values[key] for key in ("tp", "fp", "fn", "tn"))
            result[threshold] = {
                "csi": tp / (tp + fp + fn + eps),
                "pod": tp / (tp + fn + eps),
                "far": fp / (tp + fp + eps),
                "hss": 2.0 * (tp * tn - fn * fp)
                / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn) + eps),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        return result

    def compute(self) -> Dict[str, Any]:
        if self.count == 0:
            raise RuntimeError("No samples were accumulated for metrics.")
        mse = self.sse / self.count
        leads = []
        for lead in range(self.output_length):
            lead_mse = self.lead_sse[lead] / self.lead_count[lead]
            leads.append(
                {
                    "lead_index": lead,
                    "lead_minutes": (lead + 1) * 5,
                    "mse": lead_mse,
                    "rmse": math.sqrt(lead_mse),
                    "mae": self.lead_sae[lead] / self.lead_count[lead],
                    "thresholds": self._finish_counts(self.lead_counts[lead]),
                }
            )
        return {
            "mse": mse,
            "rmse": math.sqrt(mse),
            "mae": self.sae / self.count,
            "pred_min": self.pred_min,
            "pred_max": self.pred_max,
            "true_min": self.true_min,
            "true_max": self.true_max,
            "thresholds": self._finish_counts(self.counts),
            "lead_time_metrics": leads,
        }


def format_metrics(title: str, metrics: Dict[str, Any]) -> str:
    lines = [
        f"========================= {title} =========================",
        f"RMSE: {metrics['rmse']:.4f} mm/h | MAE: {metrics['mae']:.4f} mm/h",
        "Threshold (mm/h) | CSI      | POD      | FAR      | HSS",
        "------------------------------------------------------------",
    ]
    for threshold, values in metrics["thresholds"].items():
        lines.append(
            f"{threshold:<16.1f} | {values['csi']:<8.4f} | {values['pod']:<8.4f} | "
            f"{values['far']:<8.4f} | {values['hss']:<8.4f}"
        )
    lines.append(
        f"Prediction range: {metrics['pred_min']:.2f} - {metrics['pred_max']:.2f} mm/h"
    )
    lines.append(
        f"Target range: {metrics['true_min']:.2f} - {metrics['true_max']:.2f} mm/h"
    )
    return "\n".join(lines)

def average_components(totals: Dict[str, float], sample_count: int) -> Dict[str, float]:
    return {key: value / max(sample_count, 1) for key, value in totals.items()}

def add_components(criterion: nn.Module, totals: Dict[str, float], batch_size: int) -> None:
    components = getattr(criterion, "last_components", None)
    if isinstance(components, dict):
        for key, value in components.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_size

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: Dict[str, Any],
    description: str,
) -> Dict[str, Any]:
    model.eval()
    accumulator = MetricAccumulator(
        output_length=int(config["output_length"]),
        thresholds=config["metric_thresholds"],
        raw_clip_max=float(config["raw_clip_max"]),
    )
    loss_total = 0.0
    component_totals: Dict[str, float] = {}
    for inputs, targets in tqdm(loader, desc=description, ascii=True, dynamic_ncols=True):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=bool(config["use_amp"])):
            prediction, aux_loss = split_model_output(model(inputs))
            loss = criterion(prediction, targets, aux_loss)
        loss_total += float(loss.item()) * inputs.size(0)
        add_components(criterion, component_totals, inputs.size(0))
        accumulator.update(prediction, targets)
    metrics = accumulator.compute()
    metrics["loss"] = loss_total / len(loader.dataset)
    metrics["loss_components"] = average_components(component_totals, len(loader.dataset))
    return metrics

def checkpoint_paths(model_name: str, config: Dict[str, Any]) -> Dict[str, Path]:
    directory = resolve_project_path(config["checkpoint_dir"]) / model_name
    return {
        "best_loss": directory / f"best_loss_{model_name}_meteonet.pth",
        "latest": directory / f"latest_{model_name}_meteonet.pth",
    }

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_val_loss: float,
    model_name: str,
    config: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "model_name": model_name,
            "config": config,
        },
        path,
    )

def load_checkpoint(path: Path, model, optimizer=None, scheduler=None, device="cpu"):
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # older PyTorch
        state = torch.load(path, map_location=device)
    model.load_state_dict(state.get("model_state_dict", state), strict=True)
    if optimizer is not None and state.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and state.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    return state

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)

def train_one_model(
    model_name: str,
    datasets,
    config: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    train_dataset, val_dataset, test_dataset, split = datasets
    seed = int(config["seed"])
    seed_everything(seed)
    train_generator = torch.Generator().manual_seed(seed)
    train_loader = make_loader(
        train_dataset,
        int(config["batch_size"]),
        True,
        config,
        train_generator,
    )
    val_loader = make_loader(
        val_dataset,
        int(config["validation_batch_size"]),
        False,
        config,
    )
    test_loader = make_loader(
        test_dataset,
        int(config["validation_batch_size"]),
        False,
        config,
    )

    model = build_model(config).to(device)
    criterion = build_loss(config).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = optim.AdamW(
        trainable,
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )
    paths = checkpoint_paths(model_name, config)
    start_epoch = 0
    best_val_loss = float("inf")
    if bool(config["resume"]) and paths["latest"].is_file():
        state = load_checkpoint(paths["latest"], model, optimizer, scheduler, device)
        start_epoch = int(state.get("epoch", -1)) + 1
        best_val_loss = float(state.get("best_val_loss", best_val_loss))
        print(f"Resumed {model_name} from epoch {start_epoch}.")

    total_params = sum(parameter.numel() for parameter in model.parameters())
    print("\n" + "=" * 72)
    print(f"Model: {model_name}")
    print(f"Parameters: {total_params:,} ({total_params / 1e6:.3f} M)")
    print(f"Loss: {criterion.__class__.__name__}")
    print(f"Split: {split}")
    print("=" * 72)

    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["use_amp"]))
    no_improvement = 0
    history = []
    for epoch in range(start_epoch, int(config["num_epochs"])):
        train_generator.manual_seed(seed + epoch)
        model.train()
        epoch_start = time.time()
        loss_total = 0.0
        component_totals: Dict[str, float] = {}
        for inputs, targets in tqdm(
            train_loader,
            desc=f"{model_name} {epoch + 1}/{config['num_epochs']} [Train]",
            ascii=True,
            dynamic_ncols=True,
        ):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(config["use_amp"])):
                prediction, aux_loss = split_model_output(
                    forward_training_batch(model, inputs, targets)
                )
                loss = criterion(prediction, targets, aux_loss)
            scaler.scale(loss).backward()
            if float(config["gradient_clip_norm"]) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, float(config["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            loss_total += float(loss.item()) * inputs.size(0)
            add_components(criterion, component_totals, inputs.size(0))

        train_loss = loss_total / len(train_loader.dataset)
        train_components = average_components(component_totals, len(train_loader.dataset))
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            config,
            f"{model_name} {epoch + 1}/{config['num_epochs']} [Val]",
        )
        scheduler.step(val_metrics["loss"])
        elapsed = time.time() - epoch_start
        print(
            f"\nEpoch {epoch + 1}: train_loss={train_loss:.6f}, "
            f"val_loss={val_metrics['loss']:.6f}, "
            f"lr={optimizer.param_groups[0]['lr']:.3e}, time={elapsed:.1f}s"
        )
        if train_components:
            print("Train loss components: " + ", ".join(
                f"{key}={value:.6f}" for key, value in train_components.items()
            ))
        if val_metrics["loss_components"]:
            print("Val loss components: " + ", ".join(
                f"{key}={value:.6f}" for key, value in val_metrics["loss_components"].items()
            ))
        print(format_metrics("Validation", val_metrics))

        loss_improved = val_metrics["loss"] < best_val_loss
        if loss_improved:
            best_val_loss = float(val_metrics["loss"])
            no_improvement = 0
            print(f"Validation loss improved -> best={best_val_loss:.6f}; patience reset.")
        else:
            no_improvement += 1
            print(
                f"No validation loss improvement. Patience "
                f"{no_improvement}/{int(config['early_stopping_patience'])}"
            )

        save_checkpoint(
            paths["latest"], model, optimizer, scheduler, epoch,
            best_val_loss, model_name, config,
        )
        if loss_improved:
            save_checkpoint(
                paths["best_loss"], model, optimizer, scheduler, epoch,
                best_val_loss, model_name, config,
            )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "elapsed_seconds": elapsed,
            }
        )
        write_json(paths["latest"].parent / "history.json", history)
        if no_improvement >= int(config["early_stopping_patience"]):
            print(f"Early stopping: {no_improvement} epochs without lower validation loss.")
            break

    test_metrics = None
    if bool(config["evaluate_test_after_training"]):
        selected_path = paths[str(config["test_checkpoint"])]
        load_checkpoint(selected_path, model, device=device)
        test_metrics = evaluate(
            model,
            test_loader,
            criterion,
            device,
            config,
            f"{model_name} [Test:{config['test_checkpoint']}]",
        )
        print(format_metrics("Test", test_metrics))

    result = {
        "model": model_name,
        "parameters": total_params,
        "split": split,
        "best_val_loss": best_val_loss,
        "test_checkpoint": str(config["test_checkpoint"]),
        "test": test_metrics,
        "checkpoints": {key: str(value) for key, value in paths.items()},
    }
    write_json(paths["latest"].parent / "result.json", result)
    del model, optimizer, scheduler, criterion
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result

def build_model(config):
    input_length = int(config["input_length"])
    output_length = int(config["output_length"])
    image_size = int(config.get("image_size", 121))
    return HMSimVP(input_length=input_length, output_length=output_length, image_size=image_size, hid_S=32, hid_T=256, N_S=2, N_T=8, mlp_ratio=8.0, gsta_kernel_size=21, ahg_embed_channels=16, ahg_top_k=7, ahg_dropout=0.05, ahg_residual_scale=0.18, ahg_gate_bias=-0.85, overlap_eps=1e-06, encoder_expansion_ratio=int(config['encoder_rms_ds_expansion_ratio']), encoder_context_dilation=int(config['encoder_rms_ds_context_dilation']), encoder_rms_eps=float(config['encoder_rms_ds_eps']), encoder_beta_init=float(config['encoder_rms_ds_beta_init']), correction_hidden=32, correction_scale=0.25, moment_hidden=32, patch_size=8, mean_shift_limit=0.08, log_contrast_limit=0.45, gate_bias_init=2.0, return_aux=True)

def forward_training_batch(model, inputs, targets):
    return model(inputs)

def build_loss(config):
    return MCRPLoss(**{k: config['mcrp_' + k] for k in
        ('patch_size', 'base_weight', 'mean_weight', 'contrast_weight', 'log_contrast_limit')})

# Editable run settings; no dataset or checkpoint is bundled.
RUN_CONFIG = {'data_file': 'meteonet.h5',
 'checkpoint_dir': 'checkpoints_meteonet',
 'seed': 42,
 'input_length': 12,
 'output_length': 6,
 'image_size': 128,
 'validation_ratio': 0.1,
 'raw_clip_max': 833.0,
 'batch_size': 8,
 'validation_batch_size': 16,
 'num_workers': 0,
 'pin_memory': True,
 'persistent_workers': True,
 'prefetch_factor': 2,
 'train_augmentation': True,
 'learning_rate': 0.0001,
 'weight_decay': 0.0001,
 'num_epochs': 60,
 'early_stopping_patience': 8,
 'gradient_clip_norm': 0.0,
 'use_amp': True,
 'metric_thresholds': [0.1, 1.0, 5.0, 10.0, 20.0, 30.0],
 'mcrp_patch_size': 8,
 'mcrp_base_weight': 0.1,
 'mcrp_mean_weight': 0.1,
 'mcrp_contrast_weight': 0.05,
 'mcrp_log_contrast_limit': 0.45,
 'encoder_rms_ds_expansion_ratio': 1,
 'encoder_rms_ds_context_dilation': 2,
 'encoder_rms_ds_eps': 1e-06,
 'encoder_rms_ds_beta_init': 0.0,
 'resume': False,
 'evaluate_test_after_training': True,
 'test_checkpoint': 'best_loss'}

def main():
    parser = argparse.ArgumentParser(description='Train HM-SimVP on MeteoNet. H5 groups: train/images, test/images [N,18,H,W]; raw units 0.01 mm/5 min.')
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default=RUN_CONFIG['checkpoint_dir'])
    parser.add_argument('--epochs', type=int, default=RUN_CONFIG['num_epochs'])
    parser.add_argument('--workers', type=int, default=RUN_CONFIG['num_workers'])
    parser.add_argument('--batch-size', type=int, default=RUN_CONFIG['batch_size'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = dict(RUN_CONFIG, data_file=args.data, checkpoint_dir=args.output,
                  num_epochs=args.epochs, num_workers=args.workers, batch_size=args.batch_size,
                  resume=args.resume, use_amp=torch.cuda.is_available())
    train_one_model(MODEL_NAME, build_datasets(config), config,
                    torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

if __name__ == '__main__':
    main()
