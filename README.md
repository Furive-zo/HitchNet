# HitchNet

Trailer hitch angle estimation and evaluation codebase.

## Overview

이 저장소는 trailer hitch angle estimation을 위한 학습, 평가, 도메인 적응, 분석 스크립트를 포함합니다.

- 기본 학습: `scripts/train.py`
- 기본 평가: `scripts/test.py`
- 도메인 적응 학습:
  - DANN: `scripts/train_dann.py`
  - Deep CORAL: `scripts/train_coral.py`
  - CORAL-DG: `scripts/train_coral_dg.py`
- 반복 시드 학습: `scripts/train_multi_seed.sh`

실험 설정은 `configs/experiments/*.yaml` 에서 관리하고, 각 실험 config가 모델 config와 데이터셋 config를 참조합니다.

## Repository Structure

```text
configs/
  datasets/         # dataset root/split 설정
  experiments/      # 실험 단위 config
  models/           # 모델 구조 config
scripts/            # train/test/eval/plot 스크립트
models/             # 모델 구현
utils/              # config, dataset, loss, collate 유틸
ckpts/              # 학습 체크포인트 및 로그 출력
results/            # 평가 결과 출력
datasets/           # 데이터셋 위치
```

## Environment

최소한 다음 환경이 필요합니다.

- Python 3.10+
- PyTorch
- NumPy
- PyYAML
- matplotlib
- tqdm

이 저장소는 `python3 -m scripts.<name>` 형태로 실행하는 것을 기준으로 작성되어 있습니다.

## Dataset

본 저장소는 `LI-HAE` 데이터셋을 사용합니다.

- 데이터셋 DOI: `https://dx.doi.org/10.21227/ptyv-ra09`
- DOI를 통해 데이터를 받은 뒤 아래 구조에 맞게 배치하면 됩니다.

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

현재 dataset config는 아래 경로를 기준으로 되어 있습니다.

- `configs/datasets/charger.yaml`
- `configs/datasets/dummy.yaml`
- `configs/datasets/temporary.yaml`

예를 들어 `configs/datasets/charger.yaml` 은 다음 경로를 사용합니다.

- `root: datasets/LI-HAE/dataset/charger_trailer`
- `split: datasets/LI-HAE/splits/charger_trailer_split.json`

데이터를 다른 위치에 두고 싶으면 `configs/datasets/*.yaml` 의 `root`, `split` 경로만 수정하면 됩니다.

## Pretrained Weights

학습된 가중치도 다운로드해서 바로 평가에 사용할 수 있습니다.

- pretrained weights link: `https://drive.google.com/drive/folders/17NHSDGD8qk7l5HLnd8I7UzrvTcBPC9Q1?usp=sharing`

권장 배치 위치:

```text
ckpts/<experiment.name>/best.pth
```

예:

```text
ckpts/dummy_bev_resnet_regression_norm_aug1/best.pth
ckpts/charger_bev_resnet_regression_dann_dummy/best.pth
```

링크에서 받은 파일명이 다르면 원하는 실험 폴더 아래에 `best.pth` 로 두거나, 평가 시 `--ckpt` 에 실제 경로를 직접 넘기면 됩니다.

## Config System

실험은 `configs/experiments/*.yaml` 하나로 관리합니다.

예시:

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

핵심 규칙은 아래와 같습니다.

- `experiment.name`: 출력 디렉터리 이름
- `model_config`: `configs/models/` 아래 모델 설정
- `dataset_config`: `configs/datasets/` 아래 데이터셋 설정
- `train`: epoch, batch size, augmentation, normalization 등 학습 옵션
- 일부 실험은 `target_dataset_config` 를 추가로 사용하며, DANN/CORAL 계열 source-target 학습에 필요합니다

## Training

### 1. 기본 supervised 학습

```bash
python3 -m scripts.train \
  --config configs/experiments/dummy_bev_resnet_regression_norm_aug1.yaml
```

필요하면 디바이스와 worker 수를 명시할 수 있습니다.

```bash
python3 -m scripts.train \
  --config configs/experiments/dummy_bev_resnet_regression_norm_aug1.yaml \
  --device cuda \
  --num_workers 16
```

학습 결과는 기본적으로 아래에 저장됩니다.

```text
ckpts/<experiment.name>/
  best.pth
  last.pth
  best_metrics.json
  metrics_log.csv
```

### 2. DANN 학습

`target_dataset_config` 가 포함된 experiment config를 사용합니다.

```bash
python3 -m scripts.train_dann \
  --config configs/experiments/charger_bev_resnet_regression_dann_dummy.yaml
```

현재 제공된 빠른 실행 예시는 `train_hitchnet.sh` 에도 있습니다.

```bash
bash train_hitchnet.sh
```

### 3. Deep CORAL 학습

```bash
python3 -m scripts.train_coral \
  --config configs/experiments/dummy_bev_resnet_regression_coral_charger.yaml
```

### 4. CORAL-DG 학습

```bash
python3 -m scripts.train_coral_dg \
  --config configs/experiments/dummy_bev_resnet_regression_coral_dg.yaml
```

### 5. 여러 시드로 반복 학습

```bash
bash scripts/train_multi_seed.sh \
  configs/experiments/dummy_bev_resnet_regression_norm_aug2.yaml \
  43 44
```

환경변수로 실행 옵션을 제어할 수 있습니다.

```bash
DEVICE=cuda NUM_WORKERS=8 bash scripts/train_multi_seed.sh
```

## Evaluation

기본 평가는 `scripts/test.py` 로 수행합니다.

### 1. 기본 평가

```bash
python3 -m scripts.test \
  --config configs/experiments/dummy_bev_resnet_regression_norm_aug1.yaml \
  --ckpt ckpts/dummy_bev_resnet_regression_norm_aug1/best.pth
```

### 2. trailer type을 바꿔 cross-domain 평가

`--trailer_type` 을 주면 experiment config의 dataset 설정 대신 지정한 trailer dataset으로 평가합니다.

```bash
python3 -m scripts.test \
  --config configs/experiments/dummy_bev_resnet_regression_norm_aug1.yaml \
  --ckpt ckpts/dummy_bev_resnet_regression_norm_aug1/best.pth \
  --trailer_type charger \
  --num_workers 16
```

`--trailer_type` 선택 가능 값:

- `charger`
- `dummy`
- `temporary`

### 3. CSV / Plot / BEV 시각화 저장

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

대표 옵션:

- `--save_csv`: sample-level 결과 CSV 저장
- `--plot`: error distribution plot 저장
- `--plot_bins`: plot bin 크기(deg)
- `--save_err_bev`: 큰 오차 샘플의 BEV 저장
- `--err_bev_thresh`: 저장 기준 오차(deg)
- `--err_bev_max`: 저장 최대 개수
- `--save_bev_samples`: 임의 샘플 BEV 저장
- `--save_best_bev`: 최소 오차 샘플 저장
- `--save_attn`: attention overlay 저장
- `--exp_name`: 평가 결과 폴더명 override

현재 제공된 빠른 실행 예시는 `test_hitchnet.sh` 에도 있습니다.

```bash
bash test_hitchnet.sh
```

평가 결과는 보통 아래에 저장됩니다.

- 기본: `ckpts/<experiment.name>/`
- `--trailer_type` 사용 시: `results/<experiment.name>/<trailer_type>_trailer/`

## Typical Workflow

### 같은 trailer에서 학습/평가

```bash
python3 -m scripts.train \
  --config configs/experiments/dummy_bev_resnet_regression_norm_aug1.yaml

python3 -m scripts.test \
  --config configs/experiments/dummy_bev_resnet_regression_norm_aug1.yaml \
  --ckpt ckpts/dummy_bev_resnet_regression_norm_aug1/best.pth
```

### dummy에서 학습하고 charger에서 평가

```bash
python3 -m scripts.train \
  --config configs/experiments/dummy_bev_resnet_regression_norm_aug1.yaml

python3 -m scripts.test \
  --config configs/experiments/dummy_bev_resnet_regression_norm_aug1.yaml \
  --ckpt ckpts/dummy_bev_resnet_regression_norm_aug1/best.pth \
  --trailer_type charger
```

## Notes

- checkpoint 경로는 직접 지정해야 합니다. 보통 최고 성능 모델은 `ckpts/<exp_name>/best.pth` 입니다.
- 학습 결과 로그는 `metrics_log.csv`, `best_metrics.json` 으로 저장됩니다.
- 일부 config는 seed suffix를 포함한 실험명을 사용합니다. 이 경우 checkpoint 경로도 동일한 이름을 따라갑니다.
- 대용량 결과물과 체크포인트는 git에 포함하지 않는 것을 권장합니다.

## TODO

- `LI-HAE` DOI 링크 추가
- requirements 또는 environment setup 문서 분리
