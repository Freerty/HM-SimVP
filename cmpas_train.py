"""HM-SimVP CMPAS training. Dependencies: torch, numpy, h5py, tqdm."""
from __future__ import annotations
import argparse
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence
import h5py
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from model import HMSimVP, MODEL_NAME

def resolve_project_path(path):
    return str(Path(path).expanduser().resolve())

def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed all training RNGs and optionally request deterministic CUDA kernels."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(deterministic, warn_only=True)

def seed_data_loader_worker(worker_id: int) -> None:
    """Derive NumPy/Python worker seeds from the DataLoader torch seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def normalize_precipitation(data: np.ndarray, clip_max: float, clip_min: float = 0.0) -> np.ndarray:
    clipped = np.clip(data, clip_min, clip_max)
    return np.log1p(clipped).astype(np.float32) / np.log1p(clip_max)

def denormalize_precipitation(data: np.ndarray, clip_max: float) -> np.ndarray:
    restored = np.expm1(np.asarray(data, dtype=np.float32) * np.log1p(clip_max))
    return np.clip(restored, 0.0, clip_max).astype(np.float32)

def convert_mm10_to_mmh(data: np.ndarray) -> np.ndarray:
    return np.asarray(data, dtype=np.float32) * 6.0

def build_year_split_indices(
    source_years: np.ndarray,
    test_year: Optional[int] = None,
    val_year: Optional[int] = None,
) -> Dict[str, np.ndarray | List[int]]:
    unique_years = sorted(np.unique(source_years).tolist())
    if len(unique_years) < 3:
        raise ValueError("Year-based split requires at least three distinct years.")

    resolved_test_year = unique_years[-1] if test_year is None else int(test_year)
    remaining_years = [year for year in unique_years if year != resolved_test_year]
    if not remaining_years:
        raise ValueError("No years left for training/validation after choosing test year.")

    resolved_val_year = remaining_years[-1] if val_year is None else int(val_year)
    train_years = [year for year in unique_years if year not in {resolved_test_year, resolved_val_year}]
    if not train_years:
        raise ValueError("No training years left after choosing validation and test years.")

    train_indices = np.flatnonzero(np.isin(source_years, train_years))
    val_indices = np.flatnonzero(source_years == resolved_val_year)
    test_indices = np.flatnonzero(source_years == resolved_test_year)

    if train_indices.size == 0 or val_indices.size == 0 or test_indices.size == 0:
        raise ValueError("One of the train/validation/test splits is empty.")

    return {
        "train_years": train_years,
        "val_years": [resolved_val_year],
        "test_years": [resolved_test_year],
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
    }

def build_temporal_holdout_split(
    source_years: np.ndarray,
    start_indices: Optional[np.ndarray] = None,
    test_year: Optional[int] = None,
    validation_ratio: float = 0.1,
    temporal_gap_frames: int = 0,
    total_window_steps: int = 18,
) -> Dict[str, np.ndarray | List[int]]:
    unique_years = sorted(np.unique(source_years).tolist())
    if len(unique_years) < 2:
        raise ValueError("Temporal holdout split requires at least two distinct years.")

    resolved_test_year = unique_years[-1] if test_year is None else int(test_year)
    train_val_indices = np.flatnonzero(source_years != resolved_test_year)
    test_indices = np.flatnonzero(source_years == resolved_test_year)

    if train_val_indices.size == 0 or test_indices.size == 0:
        raise ValueError("Train/validation pool or test split is empty.")

    validation_ratio = float(validation_ratio)
    if not (0.0 < validation_ratio < 0.5):
        raise ValueError("validation_ratio must be between 0 and 0.5.")

    val_size = max(1, int(round(train_val_indices.size * validation_ratio)))
    if val_size >= train_val_indices.size:
        raise ValueError("validation split would consume the entire training pool.")

    train_indices = train_val_indices[:-val_size]
    val_indices = train_val_indices[-val_size:]

    if start_indices is not None and temporal_gap_frames >= 0:
        start_indices = np.asarray(start_indices)
        first_val_index = int(val_indices[0])
        first_val_year = int(source_years[first_val_index])
        first_val_start = int(start_indices[first_val_index])
        train_window_end = start_indices[train_indices] + int(total_window_steps) - 1
        same_year_as_val = source_years[train_indices] == first_val_year
        too_close_to_val = train_window_end + int(temporal_gap_frames) >= first_val_start
        keep_train = ~(same_year_as_val & too_close_to_val)
        gap_indices = train_indices[~keep_train]
        train_indices = train_indices[keep_train]
        if train_indices.size == 0:
            raise ValueError("Temporal gap removed all training samples.")
    else:
        gap_indices = np.empty((0,), dtype=train_val_indices.dtype)

    return {
        "train_years": sorted(np.unique(source_years[train_indices]).tolist()),
        "val_years": sorted(np.unique(source_years[val_indices]).tolist()),
        "test_years": [resolved_test_year],
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "gap_indices": gap_indices,
        "temporal_gap_frames": int(temporal_gap_frames),
    }

def build_sample_timestamps(
    source_year: int,
    start_index: int,
    input_steps: int,
    output_steps: int,
    step_minutes: int = 10,
) -> tuple[List[datetime], List[datetime]]:
    base_time = datetime(int(source_year), 1, 1) + timedelta(minutes=int(start_index) * step_minutes)
    input_times = [base_time + timedelta(minutes=step_minutes * offset) for offset in range(int(input_steps))]
    target_start = base_time + timedelta(minutes=step_minutes * int(input_steps))
    target_times = [target_start + timedelta(minutes=step_minutes * offset) for offset in range(int(output_steps))]
    return input_times, target_times

def read_source_years_and_starts(h5_file: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_file, "r") as handle:
        return handle["source_year"][:], handle["start_index"][:]

class CMPASSequenceDataset(Dataset):
    def __init__(
        self,
        h5_file: str,
        indices: Sequence[int],
        clip_max: float,
        normalize: bool = True,
        clip_min: float = 0.0,
    ) -> None:
        self.h5_file = h5_file
        self.indices = np.asarray(indices, dtype=np.int64)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.normalize = normalize

        with h5py.File(h5_file, "r") as handle:
            self.input_steps = int(handle.attrs["input_steps"])
            self.output_steps = int(handle.attrs["output_steps"])
            self.valid_mask = torch.as_tensor(handle["valid_mask"][:].astype(np.float32))
            self.source_year = handle["source_year"][:]
            self.start_index = handle["start_index"][:]

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        source_idx = int(self.indices[idx])
        with h5py.File(self.h5_file, "r") as handle:
            input_seq = handle["input"][source_idx].astype(np.float32)
            output_seq = handle["output"][source_idx].astype(np.float32)

        if self.normalize:
            input_seq = normalize_precipitation(input_seq, clip_max=self.clip_max, clip_min=self.clip_min)
            output_seq = normalize_precipitation(output_seq, clip_max=self.clip_max, clip_min=self.clip_min)

        return torch.from_numpy(input_seq), torch.from_numpy(output_seq)

    def get_sample_metadata(self, idx: int) -> Dict[str, object]:
        source_idx = int(self.indices[idx])
        input_times, target_times = build_sample_timestamps(
            source_year=int(self.source_year[source_idx]),
            start_index=int(self.start_index[source_idx]),
            input_steps=self.input_steps,
            output_steps=self.output_steps,
        )
        return {
            "source_index": source_idx,
            "source_year": int(self.source_year[source_idx]),
            "start_index": int(self.start_index[source_idx]),
            "input_times": input_times,
            "target_times": target_times,
        }

def split_model_output(outputs):
    if isinstance(outputs, dict):
        prediction = outputs.get("prediction")
        if prediction is None:
            raise ValueError("Model output dict must contain 'prediction'.")
        return prediction, outputs.get("aux_loss")
    if isinstance(outputs, (tuple, list)):
        return outputs[0], None
    return outputs, None

def expand_valid_mask(valid_mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    mask = valid_mask.to(device=reference.device, dtype=reference.dtype)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    return mask.expand(reference.shape[0], reference.shape[1], reference.shape[2], reference.shape[3])

class MaskedMSELoss(nn.Module):
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        aux_loss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        expanded_mask = expand_valid_mask(valid_mask, prediction)
        squared_error = (prediction - target) ** 2
        numerator = (squared_error * expanded_mask).sum()
        denominator = expanded_mask.sum().clamp_min(1.0)
        return numerator / denominator

class MCRPMaskedMSELoss(nn.Module):
    """Final MSE plus local mean/contrast supervision for MCRP."""

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
        self.reconstruction_loss = MaskedMSELoss()
        self.last_components: Dict[str, float] = {}

    def _patch_mean(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        kwargs = dict(
            kernel_size=self.patch_size,
            stride=self.patch_size,
            ceil_mode=True,
            count_include_pad=False,
        )
        numerator = F.avg_pool2d(tensor * mask, **kwargs)
        denominator = F.avg_pool2d(mask.expand_as(tensor), **kwargs).clamp_min(1e-6)
        return numerator / denominator

    def _patch_peak(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # The invalid-pixel sentinel must remain representable under AMP.
        # ``-1e6`` overflows when the incoming tensor is float16.
        invalid_value = torch.finfo(tensor.dtype).min
        masked = torch.where(
            mask.expand_as(tensor) > 0,
            tensor,
            tensor.new_full((), invalid_value),
        )
        return F.max_pool2d(
            masked,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            ceil_mode=True,
        )

    def _patch_huber(self, prediction, target, patch_mask) -> torch.Tensor:
        loss = F.smooth_l1_loss(prediction, target, reduction="none")
        weights = (patch_mask > 0).to(loss.dtype).expand_as(loss)
        return (loss * weights).sum() / weights.sum().clamp_min(1.0)

    def forward(self, prediction, target, valid_mask, aux_loss=None):
        if not isinstance(aux_loss, dict):
            raise ValueError("MCRPMaskedMSELoss requires the MCRP auxiliary dictionary.")
        base = aux_loss.get("base_prediction")
        patch_shift = aux_loss.get("patch_shift")
        patch_log_contrast = aux_loss.get("patch_log_contrast")
        if not all(isinstance(item, torch.Tensor) for item in (base, patch_shift, patch_log_contrast)):
            raise TypeError("MCRP auxiliary outputs are missing tensors.")
        pixel_mask = valid_mask.to(device=target.device, dtype=target.dtype).view(
            1, 1, target.shape[-2], target.shape[-1]
        )
        patch_mask = F.avg_pool2d(
            pixel_mask,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            ceil_mode=True,
            count_include_pad=False,
        )
        target_mean = self._patch_mean(target, pixel_mask)
        base_detached = base.detach()
        base_mean = self._patch_mean(base_detached, pixel_mask)
        target_shift = target_mean - base_mean
        target_peak = self._patch_peak(target, pixel_mask)
        base_peak = self._patch_peak(base_detached, pixel_mask)
        ratio = (target_peak - target_mean).clamp_min(1e-4) / (
            (base_peak - base_mean).clamp_min(1e-4)
        )
        target_log_contrast = ratio.log().clamp(
            -self.log_contrast_limit, self.log_contrast_limit
        )
        final_mse = self.reconstruction_loss(prediction, target, valid_mask)
        base_mse = self.reconstruction_loss(base, target, valid_mask)
        mean_loss = self._patch_huber(patch_shift, target_shift, patch_mask)
        contrast_loss = self._patch_huber(
            patch_log_contrast, target_log_contrast, patch_mask
        )
        total = (
            final_mse
            + self.base_weight * base_mse
            + self.mean_weight * mean_loss
            + self.contrast_weight * contrast_loss
        )
        self.last_components = {
            "final_mse": float(final_mse.detach().cpu()),
            "base_mse": float(base_mse.detach().cpu()),
            "mean_huber": float(mean_loss.detach().cpu()),
            "contrast_huber": float(contrast_loss.detach().cpu()),
            "total": float(total.detach().cpu()),
        }
        return total

def accumulate_loss_components(
    criterion: nn.Module,
    totals: Dict[str, float],
    batch_size: int,
) -> None:
    components = getattr(criterion, "last_components", None)
    if not isinstance(components, dict):
        return
    for name, value in components.items():
        if isinstance(value, (int, float)):
            totals[name] = totals.get(name, 0.0) + float(value) * batch_size

def average_loss_components(totals: Dict[str, float], sample_count: int) -> Dict[str, float]:
    if not totals:
        return {}
    denominator = max(1, sample_count)
    return {name: value / denominator for name, value in totals.items()}

def compute_threshold_metrics(
    outputs_mmh: np.ndarray,
    targets_mmh: np.ndarray,
    valid_mask: np.ndarray,
    thresholds: Sequence[float],
) -> Dict[float, Dict[str, float]]:
    expanded_mask = np.broadcast_to(valid_mask.astype(bool)[None, None, :, :], outputs_mmh.shape)
    masked_outputs = outputs_mmh[expanded_mask]
    masked_targets = targets_mmh[expanded_mask]
    results = {}
    eps = 1e-8

    for threshold in thresholds:
        pred_binary = masked_outputs >= threshold
        true_binary = masked_targets >= threshold
        tp = int(np.logical_and(pred_binary, true_binary).sum())
        fp = int(np.logical_and(pred_binary, ~true_binary).sum())
        fn = int(np.logical_and(~pred_binary, true_binary).sum())
        tn = int(np.logical_and(~pred_binary, ~true_binary).sum())

        csi = tp / (tp + fp + fn + eps)
        far = fp / (tp + fp + eps)
        hss = (2 * (tp * tn - fn * fp)) / (
            (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn) + eps
        )

        results[float(threshold)] = {
            "csi": float(csi),
            "far": float(far),
            "hss": float(hss),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    return results

def summarize_epoch_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    clip_max: float,
    thresholds: Sequence[float],
) -> Dict[str, object]:
    outputs_mm10 = denormalize_precipitation(outputs.detach().cpu().numpy(), clip_max=clip_max)
    targets_mm10 = denormalize_precipitation(targets.detach().cpu().numpy(), clip_max=clip_max)
    outputs_mmh = convert_mm10_to_mmh(outputs_mm10)
    targets_mmh = convert_mm10_to_mmh(targets_mm10)
    valid_np = valid_mask.detach().cpu().numpy().astype(bool)
    expanded_mask = np.broadcast_to(valid_np[None, None, :, :], outputs_mmh.shape)
    masked_outputs = outputs_mmh[expanded_mask]
    masked_targets = targets_mmh[expanded_mask]
    mse = float(np.mean((masked_outputs - masked_targets) ** 2))
    mae = float(np.mean(np.abs(masked_outputs - masked_targets)))
    rmse = float(np.sqrt(mse))
    lead_time_metrics = []
    for lead_index in range(outputs_mmh.shape[1]):
        lead_outputs = outputs_mmh[:, lead_index : lead_index + 1]
        lead_targets = targets_mmh[:, lead_index : lead_index + 1]
        lead_mask = np.broadcast_to(valid_np[None, None, :, :], lead_outputs.shape)
        lead_masked_outputs = lead_outputs[lead_mask]
        lead_masked_targets = lead_targets[lead_mask]
        lead_mse = float(np.mean((lead_masked_outputs - lead_masked_targets) ** 2))
        lead_time_metrics.append(
            {
                "lead_index": lead_index,
                "lead_minutes": (lead_index + 1) * 10,
                "mse": lead_mse,
                "mae": float(np.mean(np.abs(lead_masked_outputs - lead_masked_targets))),
                "rmse": float(np.sqrt(lead_mse)),
                "thresholds": compute_threshold_metrics(
                    lead_outputs,
                    lead_targets,
                    valid_np,
                    thresholds,
                ),
            }
        )

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "pred_min": float(np.min(masked_outputs)) if masked_outputs.size > 0 else 0.0,
        "pred_max": float(np.max(masked_outputs)) if masked_outputs.size > 0 else 0.0,
        "true_min": float(np.min(masked_targets)) if masked_targets.size > 0 else 0.0,
        "true_max": float(np.max(masked_targets)) if masked_targets.size > 0 else 0.0,
        "thresholds": compute_threshold_metrics(outputs_mmh, targets_mmh, valid_np, thresholds),
        "lead_time_metrics": lead_time_metrics,
    }

def format_metric_table(title: str, metrics: Dict[str, object]) -> str:
    lines = [
        f"========================= {title} =========================",
        f"均方根误差 (RMSE): {metrics['rmse']:.4f} mm/h",
        "--------------------------------------------------------",
        "降水阈值(mm/h)    | CSI      | FAR      | HSS     ",
        "--------------------------------------------------------",
    ]
    for threshold, values in metrics["thresholds"].items():
        lines.append(
            f"{threshold:<17.1f} | {values['csi']:<8.4f} | {values['far']:<8.4f} | {values['hss']:<8.4f}"
        )
    lines.append("========================================================")
    
    loss_val = metrics.get('loss', metrics.get('mse', 0.0))
    if isinstance(loss_val, torch.Tensor):
        loss_val = loss_val.item()
    lines.append(f"归一化空间 Loss: {loss_val:.6f}")
    lines.append(f"平均绝对误差 (MAE): {metrics['mae']:.4f} mm/h")
    
    if "pred_min" in metrics and "pred_max" in metrics:
        lines.append(f"预测值范围: {metrics['pred_min']:.2f} - {metrics['pred_max']:.2f} mm/h")
    if "true_min" in metrics and "true_max" in metrics:
        lines.append(f"真实值范围: {metrics['true_min']:.2f} - {metrics['true_max']:.2f} mm/h")
        
    return "\n".join(lines)


def build_datasets(config: Dict[str, object]):
    data_file = resolve_project_path(str(config["data_file"]))
    source_years, start_indices = read_source_years_and_starts(data_file)
    if config.get("validation_year") is not None:
        split = build_year_split_indices(
            source_years,
            test_year=config.get("test_year"),
            val_year=config.get("validation_year"),
        )
    else:
        with h5py.File(data_file, "r") as handle:
            total_window_steps = int(handle.attrs["input_steps"]) + int(handle.attrs["output_steps"])
        split = build_temporal_holdout_split(
            source_years,
            start_indices=start_indices,
            test_year=config.get("test_year"),
            validation_ratio=float(config.get("validation_ratio", 0.1)),
            temporal_gap_frames=int(config.get("temporal_gap_frames", 0)),
            total_window_steps=total_window_steps,
        )

    train_dataset = CMPASSequenceDataset(data_file, split["train_indices"], clip_max=float(config["clip_max"]))
    val_dataset = CMPASSequenceDataset(data_file, split["val_indices"], clip_max=float(config["clip_max"]))
    test_dataset = CMPASSequenceDataset(data_file, split["test_indices"], clip_max=float(config["clip_max"]))
    return train_dataset, val_dataset, test_dataset, split

def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    valid_mask: torch.Tensor,
    clip_max: float,
    thresholds: Sequence[float],
    progress_desc: Optional[str] = None,
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    loss_component_totals: Dict[str, float] = {}
    all_outputs: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    with torch.no_grad():
        iterator = data_loader
        if progress_desc:
            iterator = tqdm(data_loader, desc=progress_desc, ascii=True, dynamic_ncols=True)
        for inputs, targets in iterator:
            inputs = inputs.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)
            outputs, aux_loss = split_model_output(model(inputs))
            loss = compute_training_loss(
                criterion,
                outputs,
                targets,
                valid_mask,
                aux_loss,
                input_sequence=inputs,
            )
            total_loss += loss.item() * inputs.size(0)
            accumulate_loss_components(criterion, loss_component_totals, inputs.size(0))
            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())

    mean_loss = total_loss / max(1, len(data_loader.dataset))
    metrics = summarize_epoch_metrics(
        torch.cat(all_outputs, dim=0),
        torch.cat(all_targets, dim=0),
        valid_mask.cpu(),
        clip_max=clip_max,
        thresholds=thresholds,
    )
    metrics["loss"] = mean_loss
    loss_components = average_loss_components(loss_component_totals, len(data_loader.dataset))
    if loss_components:
        metrics["loss_components"] = loss_components
    return metrics

def build_model(config):
    input_length = int(config["input_length"])
    output_length = int(config["output_length"])
    image_size = int(config.get("image_size", 121))
    return HMSimVP(input_length=input_length, output_length=output_length, image_size=121, hid_S=32, hid_T=256, N_S=2, N_T=8, mlp_ratio=8.0, gsta_kernel_size=21, ahg_embed_channels=int(config.get('flow_ahg_embed_channels', 16)), ahg_top_k=int(config.get('flow_ahg_top_k', 7)), ahg_dropout=float(config.get('flow_ahg_dropout', 0.05)), ahg_residual_scale=float(config.get('flow_ahg_residual_scale', 0.18)), ahg_gate_bias=float(config.get('flow_ahg_gate_bias', -0.85)), overlap_eps=float(config.get('edge_interaction_overlap_eps', 1e-06)), encoder_expansion_ratio=int(config.get('encoder_rms_ds_expansion_ratio', 1)), encoder_context_dilation=int(config.get('encoder_rms_ds_context_dilation', 2)), encoder_rms_eps=float(config.get('encoder_rms_ds_eps', 1e-06)), encoder_beta_init=float(config.get('encoder_rms_ds_beta_init', 0.0)), correction_hidden=int(config.get('mcrp_correction_hidden', 32)), correction_scale=float(config.get('mcrp_residual_scale', 0.25)), moment_hidden=int(config.get('mcrp_moment_hidden', 32)), patch_size=int(config.get('mcrp_patch_size', 8)), mean_shift_limit=float(config.get('mcrp_mean_shift_limit', 0.08)), log_contrast_limit=float(config.get('mcrp_log_contrast_limit', 0.45)), gate_bias_init=float(config.get('mcrp_gate_bias_init', 2.0)), return_aux=True)

def compute_training_loss(criterion, outputs, targets, valid_mask, aux_loss, input_sequence=None):
    return criterion(outputs, targets, valid_mask, aux_loss)

def build_loss(config):
    return MCRPMaskedMSELoss(**{k: config['mcrp_' + k] for k in
        ('patch_size', 'base_weight', 'mean_weight', 'contrast_weight', 'log_contrast_limit')})

RUN_CONFIG = dict(data_file='cmpas.h5', checkpoint_dir='checkpoints_cmpas',
    input_length=12, output_length=6, batch_size=8, learning_rate=1e-4,
    weight_decay=1e-4, num_epochs=50, early_stopping_patience=8, num_workers=0,
    pin_memory=True, clip_min=0., clip_max=28., metric_thresholds=[.1,1.,5.,10.,20.,30.],
    validation_ratio=.1, validation_year=None, test_year=None, temporal_gap_frames=18,
    seed=42, deterministic=True, mcrp_patch_size=8, mcrp_base_weight=.1,
    mcrp_mean_weight=.1, mcrp_contrast_weight=.05, mcrp_log_contrast_limit=.45)

def train(config):
    set_global_seed(config['seed'], deterministic=config['deterministic'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data, val_data, _, split = build_datasets(config)
    generator = torch.Generator().manual_seed(config['seed'])
    val_generator = torch.Generator().manual_seed(config['seed'] + 1)
    def loader(data, shuffle, gen):
        return DataLoader(data, batch_size=config['batch_size'], shuffle=shuffle,
            num_workers=config['num_workers'], pin_memory=config['pin_memory'],
            worker_init_fn=seed_data_loader_worker, generator=gen)
    train_loader = loader(train_data, True, generator)
    val_loader = loader(val_data, False, val_generator)
    model = build_model(config).to(device)
    criterion = build_loss(config).to(device)
    mask = train_data.valid_mask.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=2)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    directory = Path(config['checkpoint_dir']); directory.mkdir(parents=True, exist_ok=True)
    best_loss, patience, start = float('inf'), 0, 0
    if config.get('resume'):
        state = torch.load(directory/'latest.pth', map_location=device, weights_only=False)
        model.load_state_dict(state['model_state_dict'], strict=True)
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        scaler.load_state_dict(state['scaler_state_dict'])
        best_loss = state['best_val_loss']
        start, patience = state['epoch'] + 1, state.get('patience', 0)
    print({k:v for k,v in split.items() if not isinstance(v, np.ndarray)})
    for epoch in range(start, config['num_epochs']):
        generator.manual_seed(config['seed'] + epoch)
        model.train(); total = 0.
        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]'):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                prediction, aux = split_model_output(model(inputs))
                loss = criterion(prediction, targets, mask, aux)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total += loss.item() * len(inputs)
        metrics = evaluate_model(model, val_loader, device, criterion, mask,
                                  config['clip_max'], config['metric_thresholds'])
        scheduler.step(metrics['loss'])
        loss_improved = metrics['loss'] < best_loss
        best_loss = min(best_loss, metrics['loss'])
        patience = 0 if loss_improved else patience + 1
        state = dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                     scheduler_state_dict=scheduler.state_dict(), scaler_state_dict=scaler.state_dict(),
                     epoch=epoch, best_val_loss=best_loss,
                     patience=patience, config=config, model_name=MODEL_NAME)
        torch.save(state, directory/'latest.pth')
        if loss_improved: torch.save(state, directory/'best_loss.pth')
        print(f"epoch={epoch+1} train_loss={total/len(train_data):.6f} val_loss={metrics['loss']:.6f}")
        print(format_metric_table('Validation', metrics))
        if patience >= config['early_stopping_patience']: break

def main():
    parser = argparse.ArgumentParser(description='Train HM-SimVP on CMPAS. H5: input/output [N,T,H,W] in mm/10 min; valid_mask [H,W]; source_year/start_index [N]; input_steps/output_steps attributes.')
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default=RUN_CONFIG['checkpoint_dir'])
    parser.add_argument('--epochs', type=int, default=RUN_CONFIG['num_epochs'])
    parser.add_argument('--workers', type=int, default=RUN_CONFIG['num_workers'])
    parser.add_argument('--batch-size', type=int, default=RUN_CONFIG['batch_size'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = dict(RUN_CONFIG, data_file=args.data, checkpoint_dir=args.output,
                  num_epochs=args.epochs, num_workers=args.workers, batch_size=args.batch_size,
                  resume=args.resume)
    train(config)

if __name__ == '__main__':
    main()
