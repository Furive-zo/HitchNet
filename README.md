# HitchNet

Trailer hitch angle estimation, training, evaluation, and analysis codebase.

## Overview

This repository contains training, evaluation, domain adaptation, and analysis scripts for trailer hitch angle estimation.

- Main supervised training: `scripts/train.py`
- Main evaluation: `scripts/test.py`
- Domain adaptation training:
  - DANN: `scripts/train_dann.py`
  - Deep CORAL: `scripts/train_coral.py`
  - CORAL-DG: `scripts/train_coral_dg.py`
- Multi-seed training: `scripts/train_multi_seed.sh`

Experiments are defined in `configs/experiments/*.yaml`. Each experiment config references a model config and a dataset config.

## Repository Structure

```text
configs/
  datasets/         # dataset root/split configs
  experiments/      # experiment-level configs
  models/           # model configs
scripts/            # train/test/eval/plot scripts
models/             # model implementations
utils/              # config, dataset, loss, and collate utilities
ckpts/              # training checkpoints and logs
results/            # evaluation outputs
datasets/           # local dataset location
```

## Environment

The training and evaluation runs in this repository were performed in the following conda environment:

```bash
conda activate trailer_env
```

### Create the conda environment

Use the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate trailer_env
```

If you need to update an existing environment:

```bash
conda env update -f environment.yml --prune
conda activate trailer_env
```

### Install with pip instead

For a pip-based setup, install PyTorch first according to your CUDA/CPU environment, then install the remaining packages:

```bash
pip install -r requirements.txt
```

### Core packages

The project depends on at least the following packages for standard training and evaluation:

- Python 3.10+
- PyTorch
- torchvision
- torchaudio
- NumPy
- PyYAML
- matplotlib
- SciPy
- tqdm
- Open3D
- fvcore

Notes:

- `scripts/ekf_hitch_angle.py` also depends on ROS2 packages such as `rclpy`, `sensor_msgs`, and related message packages. Those are not included in `requirements.txt` or `environment.yml`.
- Examples in this README assume commands are run after activating `trailer_env` and use `python3 -m scripts.<name>`.

## Dataset

This project uses the `LI-HAE` dataset.

- Dataset DOI: [https://dx.doi.org/10.21227/ptyv-ra09](https://dx.doi.org/10.21227/ptyv-ra09)

Trailer type mapping used in this repository:

- `dummy` = `Long-Flat`
- `charger` = `Short-Tall`
- `temporary` = `Compact`

After downloading the dataset, place it under the following structure:

```text
datasets/
  LI-HAE/
    dataset/
      charger_trailer/
      dummy_trailer/
      temporary_trailer/
    splits/
      charger_trailer_split.json
      dummy_trailer_split.json
      temporary_trailer_split.json
```

The current dataset configs are:

- `configs/datasets/charger.yaml`
- `configs/datasets/dummy.yaml`
- `configs/datasets/temporary.yaml`

For example, `configs/datasets/charger.yaml` expects:

- `root: datasets/LI-HAE/dataset/charger_trailer`
- `split: datasets/LI-HAE/splits/charger_trailer_split.json`

If your dataset is stored elsewhere, update the `root` and `split` fields in `configs/datasets/*.yaml`.

## Pretrained Weights

Pretrained checkpoints can be downloaded here:

- Weights link: [Google Drive folder](https://drive.google.com/drive/folders/17NHSDGD8qk7l5HLnd8I7UzrvTcBPC9Q1?usp=sharing)

Recommended placement:

```text
ckpts/<experiment.name>/best.pth
```

Examples:

```text
ckpts/dummy_bev_resnet_regression_norm_aug1/best.pth
ckpts/charger_bev_resnet_regression_dann_dummy/best.pth
```

If a downloaded file has a different name, either rename it to `best.pth` inside the corresponding experiment directory or pass the exact path with `--ckpt` during evaluation.

## Config System

Each experiment is controlled by a single file in `configs/experiments/*.yaml`.

Example:

```yaml
experiment:
  name: dummy_bev_resnet_regression_norm_aug1
  seed: 42

model_config: models/bev_resnet_regression_norm.yaml
dataset_config: datasets/dummy.yaml

train:
  epochs: 50
  batch_size: 512
  lr: 0.0001
```

Key fields:

- `experiment.name`: output directory name
- `model_config`: model definition under `configs/models/`
- `dataset_config`: dataset definition under `configs/datasets/`
- `train`: training hyperparameters, augmentation, normalization, and related options
- `target_dataset_config`: optional target dataset config for source-target adaptation methods such as DANN and CORAL

## Training

### 1. Standard supervised training

```bash
python3 -m scripts.train \
  --config configs/experiments/dummy_bev_resnet_regression_norm_aug1.yaml
```

You can also specify the device and worker count:

```bash
python3 -m scripts.train \
  --config configs/experiments/dummy_bev_resnet_regression_norm_aug1.yaml \
  --device cuda \
  --num_workers 16
```

Training outputs are typically written to:

```text
ckpts/<experiment.name>/
  best.pth
  last.pth
  best_metrics.json
  metrics_log.csv
```

### 2. DANN training

Use an experiment config that includes `target_dataset_config`.

```bash
python3 -m scripts.train_dann \
  --config configs/experiments/charger_bev_resnet_regression_dann_dummy.yaml
```

A quick launcher is also provided:

```bash
bash train_hitchnet.sh
```

### 3. Deep CORAL training

```bash
python3 -m scripts.train_coral \
  --config configs/experiments/dummy_bev_resnet_regression_coral_charger.yaml
```

### 4. CORAL-DG training

```bash
python3 -m scripts.train_coral_dg \
  --config configs/experiments/dummy_bev_resnet_regression_coral_dg.yaml
```

### 5. Multi-seed training

```bash
bash scripts/train_multi_seed.sh \
  configs/experiments/dummy_bev_resnet_regression_norm_aug2.yaml \
  43 44
```

You can control runtime options with environment variables:

```bash
DEVICE=cuda NUM_WORKERS=8 bash scripts/train_multi_seed.sh
```

## Evaluation

Main evaluation is done with `scripts/test.py`.

### 1. Basic evaluation

```bash
python3 -m scripts.test \
  --config configs/experiments/dummy_bev_resnet_regression_norm_aug1.yaml \
  --ckpt ckpts/dummy_bev_resnet_regression_norm_aug1/best.pth
```

### 2. Cross-domain evaluation with another trailer type

If you pass `--trailer_type`, evaluation uses that trailer dataset instead of the dataset defined in the experiment config.

```bash
python3 -m scripts.test \
  --config configs/experiments/dummy_bev_resnet_regression_norm_aug1.yaml \
  --ckpt ckpts/dummy_bev_resnet_regression_norm_aug1/best.pth \
  --trailer_type charger \
  --num_workers 16
```

Supported values for `--trailer_type`:

- `charger` (`Short-Tall`)
- `dummy` (`Long-Flat`)
- `temporary` (`Compact`)

### 3. Save CSVs, plots, and BEV visualizations

```bash
python3 -m scripts.test \
  --config configs/experiments/dummy_bev_resnet_regression_norm_aug1.yaml \
  --ckpt ckpts/dummy_bev_resnet_regression_norm_aug1/best.pth \
  --trailer_type charger \
  --save_csv \
  --plot \
  --plot_bins 5 \
  --save_err_bev \
  --err_bev_thresh 2.0 \
  --err_bev_max 50 \
  --save_best_bev
```

Useful options:

- `--save_csv`: save sample-level CSV outputs
- `--plot`: save error distribution plots
- `--plot_bins`: bin size in degrees for plots
- `--save_err_bev`: save BEV images for high-error samples
- `--err_bev_thresh`: error threshold in degrees for BEV export
- `--err_bev_max`: maximum number of high-error BEV images to save
- `--save_bev_samples`: save sample BEV images
- `--save_best_bev`: save the minimum-error sample
- `--save_attn`: save attention overlays
- `--exp_name`: override the evaluation output folder name

A quick launcher is also available:

```bash
bash test_hitchnet.sh
```

Evaluation outputs are typically written to:

- Default evaluation: `ckpts/<experiment.name>/`
- Evaluation with `--trailer_type`: `results/<experiment.name>/<trailer_type>_trailer/`

## Typical Workflows

### Train and evaluate on the same trailer type

```bash
python3 -m scripts.train \
  --config configs/experiments/dummy_bev_resnet_regression_norm_aug1.yaml

python3 -m scripts.test \
  --config configs/experiments/dummy_bev_resnet_regression_norm_aug1.yaml \
  --ckpt ckpts/dummy_bev_resnet_regression_norm_aug1/best.pth
```

### Train on `dummy` and evaluate on `charger`

```bash
python3 -m scripts.train \
  --config configs/experiments/dummy_bev_resnet_regression_norm_aug1.yaml

python3 -m scripts.test \
  --config configs/experiments/dummy_bev_resnet_regression_norm_aug1.yaml \
  --ckpt ckpts/dummy_bev_resnet_regression_norm_aug1/best.pth \
  --trailer_type charger
```

### Evaluate with a downloaded pretrained checkpoint

```bash
python3 -m scripts.test \
  --config configs/experiments/charger_bev_resnet_regression_dann_dummy.yaml \
  --ckpt ckpts/charger_bev_resnet_regression_dann_dummy/best.pth \
  --trailer_type dummy
```

## Notes

- The checkpoint path must be passed explicitly. In most cases, the best model is stored at `ckpts/<exp_name>/best.pth`.
- Training logs are saved as `metrics_log.csv` and `best_metrics.json`.
- Some experiments use seed-specific experiment names, so the checkpoint directory follows that exact name.
- Large checkpoints, datasets, and result artifacts should not be committed to git.
