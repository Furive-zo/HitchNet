# utils/load_config.py
import os
import yaml


def _load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_config(exp_cfg_path: str):
    """
    exp_cfg_path: e.g. configs/experiments/e1_charger_hitchnet.yaml

    반환값 예시:
    {
      "experiment": {...},
      "dataset": {...},
      "model": {...},
      "train": {...},
      "eval": {...},
    }
    """
    exp_cfg = _load_yaml(exp_cfg_path)

    # 상대경로 기준: experiment yaml이 있는 디렉터리
    base_dir = os.path.dirname(exp_cfg_path)

    model_cfg_path = exp_cfg["model_config"]
    dataset_cfg_path = exp_cfg["dataset_config"]

    # model_config, dataset_config가 상대경로일 경우 base_dir 기준으로 변환
    if not os.path.isabs(model_cfg_path):
        model_cfg_path = os.path.normpath(os.path.join(base_dir, "..", model_cfg_path))
    if not os.path.isabs(dataset_cfg_path):
        dataset_cfg_path = os.path.normpath(os.path.join(base_dir, "..", dataset_cfg_path))

    model_cfg = _load_yaml(model_cfg_path)
    dataset_cfg = _load_yaml(dataset_cfg_path)

    cfg = {
        "experiment": exp_cfg.get("experiment", {}),
        "train": exp_cfg.get("train", {}),
        "eval": exp_cfg.get("eval", {}),
        "model": model_cfg.get("model", {}),
        "dataset": dataset_cfg.get("dataset", {}),
    }
    return cfg
