from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from method.logging_utils import progress as tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from method import tsdb_isprs as ablation


base = ablation.base
CLASS_NAMES = list(base.CLASS_NAMES)
NUM_CLASSES = int(base.NUM_CLASSES)
IGNORE_INDEX = int(base.IGNORE_INDEX)


def ckpt_args(checkpoint: Dict) -> Dict:
    value = checkpoint.get("args", {}) or {}
    if not isinstance(value, dict):
        raise RuntimeError("Checkpoint field 'args' must be a dictionary")
    return value


def saved_arg(saved: Dict, key: str, default):
    value = saved.get(key, default)
    return default if value is None else value


def build_tsdb(
    checkpoint: Dict,
    sam_ckpt: str,
    device: torch.device,
) -> ablation.TSDB:
    saved = ckpt_args(checkpoint)
    class_weights = [
        float(value)
        for value in str(
            saved_arg(saved, "class_weights", "0.8,1.0,1.1,1.1,1.5,1.5")
        ).split(",")
    ]
    if len(class_weights) != NUM_CLASSES:
        raise RuntimeError(f"Invalid saved class weights: {class_weights}")

    use_structure = True
    model = ablation.TSDB(
        sam_ckpt=sam_ckpt,
        device=device,
        resize=int(saved_arg(saved, "resize", 512)),
        feat_channels=int(saved_arg(saved, "embed_dim", 256)),
        out_channels=int(saved_arg(saved, "embed_dim", 256)),
        num_queries=int(saved_arg(saved, "num_queries", 100)),
        decoder_layers=int(saved_arg(saved, "decoder_layers", 6)),
        num_heads=int(saved_arg(saved, "num_heads", 8)),
        use_gse=True,
        query_mode=str(saved_arg(saved, "query_mode", "add")),
        gse_base_channels=int(saved_arg(saved, "gse_base_channels", 64)),
        replace_query_feat=bool(saved_arg(saved, "replace_query_feat", False)),
        no_object_weight=float(saved_arg(saved, "no_object_weight", 0.1)),
        class_weights=class_weights,
        use_pixel_aux=bool(saved_arg(saved, "use_pixel_aux", False)),
        lambda_pixel=float(saved_arg(saved, "lambda_pixel", 0.4)),
        pixel_fusion_weight=float(saved_arg(saved, "pixel_fusion_weight", 0.5)),
        use_earth_adapter=True,
        earth_dim=int(saved_arg(saved, "earth_dim", 24)),
        earth_adapter_start=int(saved_arg(saved, "earth_adapter_start", 0)),
        earth_fft_start=int(saved_arg(saved, "earth_fft_start", 24)),
        earth_cutoff_ratio=float(saved_arg(saved, "earth_cutoff_ratio", 0.3)),
        earth_scale=float(saved_arg(saved, "earth_scale", 0.1)),
        use_structure_injection=use_structure,
        structure_scale=float(saved_arg(saved, "structure_scale", 0.05)),
    ).to(device)

    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    bad_missing = [key for key in missing if not key.startswith("sam_model.")]
    if bad_missing or unexpected:
        raise RuntimeError(
            "TSDB state mismatch: "
            f"bad_missing={bad_missing[:20]}, unexpected={unexpected[:20]}"
        )
    model.eval()
    model.sam_model.eval()
    return model
