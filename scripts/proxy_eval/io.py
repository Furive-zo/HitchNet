#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""I/O helpers for proxy evaluation logs and config matching."""

from __future__ import annotations

import csv
import fnmatch
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


CANON_COLS = (
    "t",
    "v",
    "yaw_rate",
    "gamma_gt",
    "gamma_pred",
    "x",
    "y",
    "yaw",
    "frame_dir",
    "imu_wx",
    "imu_wy",
    "imu_wz",
)


@dataclass(frozen=True)
class TrailerParams:
    """Kinematic constants for one sequence."""

    L2: float
    eps: float
    matched_rule: str = "cli_default"


def parse_colmap(spec: str) -> Dict[str, str]:
    """
    Parse column mapping spec: "canon:actual,canon2:actual2".

    Returns dict canonical_name -> csv_column_name.
    """
    mapping = {k: k for k in CANON_COLS}
    if not spec:
        return mapping
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Invalid --colmap token: {token!r}. Use canon:actual format.")
        canon, actual = [x.strip() for x in token.split(":", 1)]
        if canon not in mapping:
            raise ValueError(f"Unknown canonical column in --colmap: {canon!r}")
        mapping[canon] = actual
    return mapping


def expand_input_glob(input_glob: str) -> List[str]:
    """Expand glob into sorted file list; raise if empty."""
    files = sorted(glob.glob(input_glob))
    if not files:
        raise FileNotFoundError(f"No input files matched: {input_glob}")
    return files


def load_trailer_rules(config_json: Optional[str]) -> Dict:
    """Load trailer parameter matching rules from JSON. Empty dict if None."""
    if not config_json:
        return {}
    with open(config_json, "r") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("--config_json must contain a JSON object.")
    return cfg


def match_trailer_params(
    file_path: str,
    default_L2: float,
    default_eps: float,
    rule_cfg: Optional[Dict],
) -> TrailerParams:
    """
    Match L2/eps for file using config rules.

    Supported schema:
      {
        "default": {"L2": 6.2, "eps": 0.0},
        "rules": [
          {"pattern": "*charger*", "L2": 6.2, "eps": 0.0, "name": "charger"},
          {"pattern": "*dummy*", "L2": 5.4, "eps": 0.0}
        ]
      }
    Pattern is matched against both full path and basename.
    """
    if not rule_cfg:
        return TrailerParams(L2=float(default_L2), eps=float(default_eps), matched_rule="cli_default")

    d = rule_cfg.get("default", {})
    l2 = float(d.get("L2", default_L2))
    eps = float(d.get("eps", default_eps))
    matched = "default"

    for r in rule_cfg.get("rules", []):
        if not isinstance(r, dict):
            continue
        pat = str(r.get("pattern", "")).strip()
        if not pat:
            continue
        base = os.path.basename(file_path)
        if fnmatch.fnmatch(file_path, pat) or fnmatch.fnmatch(base, pat):
            l2 = float(r.get("L2", l2))
            eps = float(r.get("eps", eps))
            matched = str(r.get("name", pat))
            break
    return TrailerParams(L2=l2, eps=eps, matched_rule=matched)


def _read_csv_as_dict_arrays(csv_path: str) -> Dict[str, np.ndarray]:
    """Read CSV into dict[str, np.ndarray(float64)], keeping missing as nan."""
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        fieldnames = list(reader.fieldnames)
        cols = {k: [] for k in fieldnames}
        for row in reader:
            for k in fieldnames:
                v = row.get(k, "")
                if v is None or v == "":
                    cols[k].append(np.nan)
                else:
                    try:
                        cols[k].append(float(v))
                    except ValueError:
                        cols[k].append(np.nan)
    return {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}


def load_log_file(path: str, colmap: Dict[str, str], gamma_unit: str, require_yaw_rate: bool = True) -> Dict[str, np.ndarray]:
    """
    Load one sequence log.

    Required canonical fields: t, v, yaw_rate, gamma_gt, gamma_pred.
    Optional canonical fields: x, y, yaw.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext != ".csv":
        raise ValueError(f"Unsupported extension {ext!r} for now. Expected CSV.")

    raw = _read_csv_as_dict_arrays(path)

    def get_col(canon: str, required: bool) -> Optional[np.ndarray]:
        name = colmap[canon]
        if name in raw:
            return raw[name]
        if required:
            raise KeyError(f"Missing required column {name!r} (canonical: {canon!r}) in {path}")
        return None

    out: Dict[str, np.ndarray] = {
        "t": get_col("t", True),
        "v": get_col("v", True),
        "yaw_rate": get_col("yaw_rate", require_yaw_rate),
        "gamma_gt": get_col("gamma_gt", True),
        "gamma_pred": get_col("gamma_pred", True),
    }
    if out["yaw_rate"] is None:
        out["yaw_rate"] = np.full_like(out["t"], np.nan, dtype=np.float64)

    x = get_col("x", False)
    y = get_col("y", False)
    yaw = get_col("yaw", False)
    if x is not None and y is not None and yaw is not None:
        out["x"] = x
        out["y"] = y
        out["yaw"] = yaw

    if gamma_unit.lower() == "deg":
        out["gamma_gt"] = np.deg2rad(out["gamma_gt"])
        out["gamma_pred"] = np.deg2rad(out["gamma_pred"])
    elif gamma_unit.lower() != "rad":
        raise ValueError(f"--gamma_unit must be 'rad' or 'deg', got {gamma_unit!r}")

    n = len(out["t"])
    for k, v in out.items():
        if len(v) != n:
            raise ValueError(f"Column length mismatch in {path}: {k}")
    return out


def parse_rotation_matrix(spec: str) -> np.ndarray:
    """Parse 3x3 rotation matrix from 'a,b,c,d,e,f,g,h,i'."""
    vals = [float(x.strip()) for x in spec.split(",") if x.strip()]
    if len(vals) != 9:
        raise ValueError("--imu_rotation must contain 9 comma-separated values.")
    return np.asarray(vals, dtype=np.float64).reshape(3, 3)


def _json_scalar(value: object) -> float:
    """Convert scalar/list JSON field into one float (median for list-like)."""
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return float(np.nan)
        arr = np.asarray(value, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float(np.nan)
        return float(np.median(arr))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(np.nan)


def override_yaw_rate_from_imu_csv(
    log: Dict[str, np.ndarray],
    raw_path: str,
    colmap: Dict[str, str],
    R_flu_from_rt: np.ndarray,
) -> None:
    """
    Replace log['yaw_rate'] using imu wx/wy/wz columns from the same CSV.
    yaw_rate := (R_flu_from_rt @ [wx, wy, wz])[2].
    """
    raw = _read_csv_as_dict_arrays(raw_path)
    wx_key = colmap["imu_wx"]
    wy_key = colmap["imu_wy"]
    wz_key = colmap["imu_wz"]
    if wx_key not in raw or wy_key not in raw or wz_key not in raw:
        raise KeyError(
            "imu_csv source requires imu_wx/imu_wy/imu_wz via --colmap, "
            f"missing among ({wx_key}, {wy_key}, {wz_key})"
        )
    w_rt = np.stack([raw[wx_key], raw[wy_key], raw[wz_key]], axis=0).astype(np.float64)
    if w_rt.shape[1] != log["t"].shape[0]:
        raise ValueError("IMU angular-velocity column length mismatch with main series.")
    w_flu = R_flu_from_rt @ w_rt
    log["yaw_rate"] = w_flu[2]


def override_yaw_rate_from_vehicle_imu_json(
    log: Dict[str, np.ndarray],
    raw_path: str,
    colmap: Dict[str, str],
    R_flu_from_rt: np.ndarray,
    imu_frame_base_dir: str = "",
) -> None:
    """
    Replace log['yaw_rate'] by reading per-row <frame_dir>/vehicle_imu.json.

    frame_dir can be absolute or relative to imu_frame_base_dir.
    """
    raw = _read_csv_as_dict_arrays(raw_path)
    frame_key = colmap["frame_dir"]
    if frame_key not in raw:
        raise KeyError(
            "vehicle_imu_json source requires frame_dir column via --colmap, "
            f"missing: {frame_key!r}"
        )

    # Re-read strings from CSV because numeric loader stores NaN.
    frame_dirs: List[str] = []
    with open(raw_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_dirs.append((row.get(frame_key, "") or "").strip())

    n = log["t"].shape[0]
    if len(frame_dirs) != n:
        raise ValueError("frame_dir length mismatch with main series.")

    cache: Dict[str, float] = {}
    yaw = np.full(n, np.nan, dtype=np.float64)
    for i, d in enumerate(frame_dirs):
        if not d:
            continue
        if not os.path.isabs(d):
            d = os.path.join(imu_frame_base_dir, d) if imu_frame_base_dir else d
        if d in cache:
            yaw[i] = cache[d]
            continue
        jpath = os.path.join(d, "vehicle_imu.json")
        if not os.path.exists(jpath):
            raise FileNotFoundError(f"vehicle_imu.json not found: {jpath}")
        with open(jpath, "r") as f:
            js = json.load(f)
        wx = _json_scalar(js.get("angular_velocity_x", np.nan))
        wy = _json_scalar(js.get("angular_velocity_y", np.nan))
        wz = _json_scalar(js.get("angular_velocity_z", np.nan))
        w_flu = R_flu_from_rt @ np.array([wx, wy, wz], dtype=np.float64)
        yr = float(w_flu[2])
        cache[d] = yr
        yaw[i] = yr

    log["yaw_rate"] = yaw
