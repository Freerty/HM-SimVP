"""CMPAS pooled and lead-time evaluation; no plotting or baselines."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from cmpas_train import (RUN_CONFIG, build_model, build_datasets, MaskedMSELoss,
    split_model_output, denormalize_precipitation, convert_mm10_to_mmh)
def tensor_to_mmh(data: torch.Tensor, clip_max: float) -> np.ndarray:
    data_mm10 = denormalize_precipitation(data.detach().cpu().numpy(), clip_max=clip_max)
    return convert_mm10_to_mmh(data_mm10)

def expand_valid_mask_np(valid_mask: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if valid_mask.ndim != 2:
        raise ValueError(f"valid_mask must be 2D, got shape {valid_mask.shape}")
    if reference.ndim == 4:
        return np.broadcast_to(valid_mask.astype(bool)[None, None, :, :], reference.shape)
    if reference.ndim == 3:
        return np.broadcast_to(valid_mask.astype(bool)[None, :, :], reference.shape)
    raise ValueError(f"Unsupported reference ndim: {reference.ndim}")

def compute_threshold_metrics_with_pod(
    outputs_mmh: np.ndarray,
    targets_mmh: np.ndarray,
    valid_mask: np.ndarray,
    thresholds: List[float],
) -> Dict[float, Dict[str, float]]:
    expanded_mask = expand_valid_mask_np(valid_mask, outputs_mmh)
    masked_outputs = outputs_mmh[expanded_mask]
    masked_targets = targets_mmh[expanded_mask]
    results: Dict[float, Dict[str, float]] = {}
    eps = 1e-8

    for threshold in thresholds:
        pred_binary = masked_outputs >= threshold
        true_binary = masked_targets >= threshold
        tp = int(np.logical_and(pred_binary, true_binary).sum())
        fp = int(np.logical_and(pred_binary, ~true_binary).sum())
        fn = int(np.logical_and(~pred_binary, true_binary).sum())
        tn = int(np.logical_and(~pred_binary, ~true_binary).sum())

        csi = tp / (tp + fp + fn + eps)
        pod = tp / (tp + fn + eps)
        far = fp / (tp + fp + eps)
        hss = (2 * (tp * tn - fn * fp)) / (
            (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn) + eps
        )

        results[float(threshold)] = {
            "csi": float(csi),
            "pod": float(pod),
            "far": float(far),
            "hss": float(hss),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    return results

def summarize_test_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    clip_max: float,
    thresholds: List[float],
) -> Dict[str, object]:
    outputs_mm10 = denormalize_precipitation(outputs.detach().cpu().numpy(), clip_max=clip_max)
    targets_mm10 = denormalize_precipitation(targets.detach().cpu().numpy(), clip_max=clip_max)
    outputs_mmh = convert_mm10_to_mmh(outputs_mm10)
    targets_mmh = convert_mm10_to_mmh(targets_mm10)
    valid_np = valid_mask.detach().cpu().numpy().astype(bool)
    expanded_mask = expand_valid_mask_np(valid_np, outputs_mmh)
    masked_outputs = outputs_mmh[expanded_mask]
    masked_targets = targets_mmh[expanded_mask]
    mse = float(np.mean((masked_outputs - masked_targets) ** 2))
    mae = float(np.mean(np.abs(masked_outputs - masked_targets)))
    rmse = float(np.sqrt(mse))

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "pred_min": float(np.min(masked_outputs)) if masked_outputs.size > 0 else 0.0,
        "pred_max": float(np.max(masked_outputs)) if masked_outputs.size > 0 else 0.0,
        "true_min": float(np.min(masked_targets)) if masked_targets.size > 0 else 0.0,
        "true_max": float(np.max(masked_targets)) if masked_targets.size > 0 else 0.0,
        "thresholds": compute_threshold_metrics_with_pod(outputs_mmh, targets_mmh, valid_np, thresholds),
    }

def evaluate_test_model_with_lead_times(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: MaskedMSELoss,
    valid_mask: torch.Tensor,
    clip_max: float,
    thresholds: List[float],
    lead_step_minutes: int = 10,
    progress_desc: str | None = None,
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    all_outputs: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    with torch.no_grad():
        iterator = data_loader
        if progress_desc:
            iterator = tqdm(data_loader, desc=progress_desc, ascii=True, dynamic_ncols=True)
        for inputs, targets in iterator:
            # ``Tensor.to(device)`` preserves the source dtype.  Some external
            # CMPAS H5 files are loaded as float64, whereas all evaluation
            # models/checkpoints use float32 parameters.  Cast both tensors at
            # the evaluation boundary so a double batch cannot reach Conv2d.
            inputs = inputs.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)
            outputs, _ = split_model_output(model(inputs))
            loss = criterion(outputs, targets, valid_mask)
            total_loss += loss.item() * inputs.size(0)
            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())

    outputs_tensor = torch.cat(all_outputs, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)
    overall_metrics = summarize_test_metrics(
        outputs=outputs_tensor,
        targets=targets_tensor,
        valid_mask=valid_mask.cpu(),
        clip_max=clip_max,
        thresholds=thresholds,
    )
    overall_metrics["loss"] = total_loss / max(1, len(data_loader.dataset))

    lead_time_metrics: List[Dict[str, object]] = []
    for lead_index in range(outputs_tensor.shape[1]):
        lead_outputs = outputs_tensor[:, lead_index : lead_index + 1, ...]
        lead_targets = targets_tensor[:, lead_index : lead_index + 1, ...]
        lead_metrics = summarize_test_metrics(
            outputs=lead_outputs,
            targets=lead_targets,
            valid_mask=valid_mask.cpu(),
            clip_max=clip_max,
            thresholds=thresholds,
        )
        lead_metrics["lead_index"] = lead_index
        lead_metrics["lead_minutes"] = (lead_index + 1) * lead_step_minutes
        lead_time_metrics.append(lead_metrics)

    return {
        "overall": overall_metrics,
        "lead_time_metrics": lead_time_metrics,
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate HM-SimVP on held-out CMPAS data; export pooled and six lead-time metrics.')
    parser.add_argument('--data', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', default='cmpas_metrics.json')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=0)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = dict(RUN_CONFIG); config.update(state.get('config', {})); config['data_file'] = args.data
    config['metric_thresholds'] = [.1,1.,5.,10.,20.,30.]
    model = build_model(config).to(device)
    model.load_state_dict(state.get('model_state_dict', state), strict=True)
    _, _, dataset, _ = build_datasets(config)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
    result = evaluate_test_model_with_lead_times(model, loader, device, MaskedMSELoss(),
        dataset.valid_mask.to(device), config['clip_max'], config['metric_thresholds'], progress_desc='CMPAS test')
    result['checkpoint'] = str(Path(args.checkpoint).resolve())
    result['samples'] = len(dataset)
    dest = Path(args.output); dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
