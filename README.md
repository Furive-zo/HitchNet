# HitchNet

Hitch angle estimation codebase for the `LI-HAE` trailer dataset.

## Highlights

- Supervised training: `scripts/train.py`
- Evaluation: `scripts/test.py`
- Domain adaptation: `scripts/train_dann.py`, `scripts/train_coral.py`, `scripts/train_coral_dg.py`
- Multi-seed training: `scripts/train_multi_seed.sh`

Trailer type mapping:

- `dummy` = `Long-Flat`
- `charger` = `Short-Tall`
- `temporary` = `Compact`

## Setup

Training and evaluation were run in the `conda` environment `trailer_env`.

Create it with:

```bash
conda env create -f environment.yml
conda activate trailer_env
```

Update an existing environment with:

```bash
conda env update -f environment.yml --prune
conda activate trailer_env
```

If you prefer `pip`, install PyTorch for your CUDA or CPU environment first, then run:

```bash
pip install -r requirements.txt
```

Core packages:

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

`scripts/ekf_hitch_angle.py` additionally requires ROS2 packages and is not covered by `environment.yml`.

## Dataset

- Dataset DOI: [LI-HAE](https://dx.doi.org/10.21227/ptyv-ra09)
- Pretrained weights: [Google Drive folder](https://drive.google.com/drive/folders/17NHSDGD8qk7l5HLnd8I7UzrvTcBPC9Q1?usp=sharing)

Expected local layout:

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

Dataset configs live in:

- `configs/datasets/charger.yaml`
- `configs/datasets/dummy.yaml`
- `configs/datasets/temporary.yaml`

If your dataset is stored elsewhere, update `root` and `split` in the dataset config files.

Recommended pretrained checkpoint placement:

```text
ckpts/<experiment.name>/best.pth
```

## Configs

Experiments are defined in `configs/experiments/*.yaml`.

Each experiment config points to:

- a model config in `configs/models/`
- a dataset config in `configs/datasets/`
- training options under `train`

For adaptation methods such as DANN and CORAL, use configs that also include `target_dataset_config`.

## Training

Standard supervised training:

```bash
python3 -m scripts.train \
  --config configs/experiments/dummy_bev_resnet_regression_norm_aug1.yaml
```

With explicit device and worker count:

```bash
python3 -m scripts.train \
  --config configs/experiments/dummy_bev_resnet_regression_norm_aug1.yaml \
  --device cuda \
  --num_workers 16
```

DANN:

```bash
python3 -m scripts.train_dann \
  --config configs/experiments/charger_bev_resnet_regression_dann_dummy.yaml
```

Deep CORAL:

```bash
python3 -m scripts.train_coral \
  --config configs/experiments/dummy_bev_resnet_regression_coral_charger.yaml
```

CORAL-DG:

```bash
python3 -m scripts.train_coral_dg \
  --config configs/experiments/dummy_bev_resnet_regression_coral_dg.yaml
```

Multi-seed:

```bash
bash scripts/train_multi_seed.sh \
  configs/experiments/dummy_bev_resnet_regression_norm_aug2.yaml \
  43 44
```

Training outputs are written to:

```text
ckpts/<experiment.name>/
  best.pth
  last.pth
  best_metrics.json
  metrics_log.csv
```

## Evaluation

Basic evaluation:

```bash
python3 -m scripts.test \
  --config configs/experiments/dummy_bev_resnet_regression_norm_aug1.yaml \
  --ckpt ckpts/dummy_bev_resnet_regression_norm_aug1/best.pth
```

Cross-domain evaluation:

```bash
python3 -m scripts.test \
  --config configs/experiments/dummy_bev_resnet_regression_norm_aug1.yaml \
  --ckpt ckpts/dummy_bev_resnet_regression_norm_aug1/best.pth \
  --trailer_type charger \
  --num_workers 16
```

Supported `--trailer_type` values:

- `charger` (`Short-Tall`)
- `dummy` (`Long-Flat`)
- `temporary` (`Compact`)

Save CSVs and plots:

```bash
python3 -m scripts.test \
  --config configs/experiments/dummy_bev_resnet_regression_norm_aug1.yaml \
  --ckpt ckpts/dummy_bev_resnet_regression_norm_aug1/best.pth \
  --trailer_type charger \
  --save_csv \
  --plot \
  --plot_bins 5
```

Evaluation outputs are written to:

- `ckpts/<experiment.name>/`
- `results/<experiment.name>/<trailer_type>_trailer/` when `--trailer_type` is used

## License

This project is licensed under the Apache License 2.0. See [LICENSE](/home/future/furive-zo/HitchNet/LICENSE).
