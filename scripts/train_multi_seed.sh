#!/usr/bin/env bash
set -euo pipefail

# Train additional seeds by generating
# per-seed temporary configs that override experiment.name and experiment.seed.

BASE_CONFIG="configs/experiments/dummy_bev_resnet_regression_norm_aug2.yaml"
DEVICE="${DEVICE:-cuda}"
# Let the config control data loading by default; override if needed.
# Example: NUM_WORKERS=8 bash scripts/train_multi_seed.sh
NUM_WORKERS="${NUM_WORKERS:-}"
# Runtime safety toggles for environments with CUDA/cuDNN ABI mismatch.
# Set to 1 to force-disable AMP or cuDNN; defaults are speed-optimized.
FORCE_NO_AMP="${FORCE_NO_AMP:-0}"
DISABLE_CUDNN="${DISABLE_CUDNN:-0}"

# If the first arg is a yaml, use it as base config.
if [ "$#" -ge 1 ] && [[ "$1" == *.yaml ]]; then
  BASE_CONFIG="$1"
  shift
fi

# Default: two additional seeds beyond 42.
if [ "$#" -ge 1 ]; then
  SEEDS=("$@")
else
  SEEDS=(43 44)
fi

if [ ! -f "$BASE_CONFIG" ]; then
  echo "[ERROR] Base config not found: $BASE_CONFIG" >&2
  exit 1
fi

BASE_CONFIG_DIR="$(cd "$(dirname "$BASE_CONFIG")" && pwd)"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
TMP_FILES=()
cleanup() {
  for f in "${TMP_FILES[@]:-}"; do
    [ -f "$f" ] && rm -f "$f"
  done
}
trap cleanup EXIT

echo "[INFO] Base config: $BASE_CONFIG"
echo "[INFO] Device: $DEVICE"
echo "[INFO] Seeds: ${SEEDS[*]}"
echo "[INFO] FORCE_NO_AMP: $FORCE_NO_AMP"
echo "[INFO] DISABLE_CUDNN: $DISABLE_CUDNN"

for seed in "${SEEDS[@]}"; do
  echo "[INFO] Preparing seed=${seed}"
  CFG_OUT="${BASE_CONFIG_DIR}/.tmp_charger_bev_resnet_regression_seed${seed}_${RUN_TAG}.yaml"
  TMP_FILES+=("$CFG_OUT")

  python3 - "$BASE_CONFIG" "$CFG_OUT" "$seed" <<'PY'
import os
import sys
import yaml

base_cfg, out_cfg, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(base_cfg, "r") as f:
    cfg = yaml.safe_load(f)

exp = cfg.setdefault("experiment", {})
base_name = exp.get("name", "charger_bev_resnet_regression")
exp["seed"] = seed
exp["name"] = f"{base_name}_seed{seed}"
train = cfg.setdefault("train", {})
force_no_amp = os.environ.get("FORCE_NO_AMP", "1") == "1"
if force_no_amp:
    train["amp"] = False

with open(out_cfg, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

  echo "[INFO] Launch train: seed=${seed}"
  TRAIN_ARGS=(--config "$CFG_OUT" --device "$DEVICE")
  if [ -n "${NUM_WORKERS}" ]; then
    TRAIN_ARGS+=(--num_workers "$NUM_WORKERS")
  fi
  if [ "$DISABLE_CUDNN" = "1" ]; then
    HITCHNET_DISABLE_CUDNN=1 FORCE_NO_AMP="$FORCE_NO_AMP" \
      python3 -m scripts.train "${TRAIN_ARGS[@]}"
  else
    FORCE_NO_AMP="$FORCE_NO_AMP" \
      python3 -m scripts.train "${TRAIN_ARGS[@]}"
  fi
done
