# HM-SimVP

Standalone PyTorch implementation for CMPAS and MeteoNet precipitation forecasting.

The training scripts read preprocessed MeteoNet HDF5 sequences; spatial cropping, quality control, missing-value treatment, and sequence construction are performed upstream as described in the manuscript.

## Files

- `model.py`: all model components and the complete HM-SimVP architecture.
- `cmpas_train.py` / `cmpas_test.py`: CMPAS training and evaluation.
- `meteonet_train.py` / `meteonet_test.py`: MeteoNet training and evaluation.

No imports from the original project, ablation models, or comparison models are required. Data and pretrained weights are not bundled.

## Environment

Python 3.11 and PyTorch 2.3 or later. Install a PyTorch build appropriate for your CPU/CUDA environment, then install the remaining dependencies:

```bash
pip install numpy h5py tqdm
```

Run commands below from this directory. CPU is supported; CUDA is used automatically when available. Editable training settings are in `RUN_CONFIG` in each training file. Relative paths resolve against the working directory.

## Model and training

Input: `[B, 12, H, W]`; output: `[B, 6, H, W]`.
CMPAS uses 10-minute frames (120-minute history, 60-minute forecast); MeteoNet uses 5-minute frames (60-minute history, 30-minute forecast).

```python
import torch
from model import HMSimVP

model = HMSimVP(image_size=121, return_aux=False).eval()
with torch.no_grad():
    forecast = model(torch.rand(1, 12, 121, 121))
```

The training entries enable auxiliary outputs for the MCRP objective. Training uses AdamW and a validation-loss-driven ReduceLROnPlateau scheduler. The validation objective includes the final reconstruction and auxiliary losses; it is not just the final-output MSE. The checkpoint with the lowest validation loss is saved as `best_loss`; early stopping also monitors validation loss. `latest` supports resuming. There is no composite metric-based checkpoint selection in this release.

These commands describe this standalone release. They do not establish that a newly trained checkpoint reproduces the manuscript's reported numerical results.

## CMPAS data

Provide an HDF5 file with:

| Entry | Shape / meaning |
|---|---|
| `input` | `[N,12,H,W]`, precipitation in mm accumulated over 10 minutes |
| `output` | `[N,6,H,W]`, same units |
| `valid_mask` | `[H,W]`, 1 for valid cells, 0 outside the valid region |
| `source_year` | `[N]`, source year of each sample |
| `start_index` | `[N]`, first input frame's index within its source year |
| attribute `input_steps` | 12 |
| attribute `output_steps` | 6 |

Samples must be ordered chronologically. Use finite values in input/output arrays, including masked cells. By default, the latest source year is held out for testing; the final 10% of the remaining samples forms validation. Training windows too close to validation are removed using the configured 18-frame temporal gap. An explicit validation year can instead be set in `RUN_CONFIG`.

Values are clipped to [0,28] mm/10 min and transformed as `log1p(value)/log1p(28)`. Evaluation reverses this transform and multiplies by 6 to obtain mm/h, using the valid-region mask.

```bash
python cmpas_train.py --data /path/to/cmpas.h5 --output checkpoints_cmpas
python cmpas_test.py --data /path/to/cmpas.h5 --checkpoint checkpoints_cmpas/best_loss.pth --output cmpas_metrics.json
```

Resume using the same training settings and output directory:

```bash
python cmpas_train.py --data /path/to/cmpas.h5 --output checkpoints_cmpas --resume
```

## MeteoNet data

Provide HDF5 datasets `train/images` and `test/images`, each shaped `[N,T,H,W]`, with `T >= 18`. The standard spatial size is 128×128. Values are raw accumulation units of **0.01 mm per 5 minutes**, not mm/h. The first 12 frames are inputs and the next 6 are targets.

The final 10% of `train/images` is used for validation; keep samples chronologically ordered and prepare the train/test groups before running. Augmentation applies only to training. The loader clips negative raw values to zero and uses `log1p(value)/log1p(833)`; despite the configuration name `raw_clip_max`, this loader does not upper-clip raw values at 833. Evaluation converts raw units to mm/h by multiplying by 0.12.

```bash
python meteonet_train.py --data /path/to/meteonet.h5 --output checkpoints_meteonet
python meteonet_test.py --data /path/to/meteonet.h5 --checkpoint checkpoints_meteonet/simvp2_original_rms_ds_edge_ahg_mcrp_rgate/best_loss_simvp2_original_rms_ds_edge_ahg_mcrp_rgate_meteonet.pth --output meteonet_metrics.json
```

By default, MeteoNet training also evaluates its lowest-validation-loss checkpoint after training. Set `evaluate_test_after_training=False` in `RUN_CONFIG` to run evaluation separately. Add `--resume` to the training command to resume from `latest`.

## Evaluation and options

Both test entries accept an explicit `--checkpoint` and export pooled and six-lead-time results to JSON, including RMSE, MAE, CSI, POD, FAR and HSS at 0.1, 1, 5, 10, 20 and 30 mm/h. Categorical scores pool contingency counts; they are not averages of per-image scores. Checkpoint configuration is loaded where present; the command-line data path overrides its stored path.

Common training options: `--epochs`, `--batch-size`, `--workers`, `--output`, `--resume`.
Common testing options: `--batch-size`, `--workers`, `--output`.
Use `python <script>.py --help` for details. Only load checkpoints from trusted sources.

