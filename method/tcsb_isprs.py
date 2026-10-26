import argparse
import hashlib
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from method.logging_utils import progress as tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


CLASS_NAMES = [
    "impervious_surface",
    "building",
    "low_vegetation",
    "tree",
    "car",
    "clutter_background",
]

PROMPT_CLASS_NAMES = [
    "impervious surface",
    "building",
    "low vegetation",
    "tree",
    "car",
]

NUM_CLASSES = 6
IGNORE_INDEX = 255


def set_seed(seed=None):
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def remap_isprs_raw_gt(raw_mask: np.ndarray) -> np.ndarray:

    raw_mask = np.asarray(raw_mask)

    if raw_mask.ndim == 3:
        raw_mask = raw_mask[..., 0]

    mapped = np.full(raw_mask.shape, IGNORE_INDEX, dtype=np.int64)
    valid = (raw_mask >= 0) & (raw_mask < NUM_CLASSES)
    mapped[valid] = raw_mask[valid].astype(np.int64)
    mapped[raw_mask == IGNORE_INDEX] = IGNORE_INDEX
    return mapped


class ISPRSSingleImageDataset(Dataset):
    def __init__(self, img_dir, ann_dir, resize=512, max_items=None):
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.resize = resize

        if not self.img_dir.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {self.img_dir}")
        if not self.ann_dir.is_dir():
            raise FileNotFoundError(f"Annotation directory does not exist: {self.ann_dir}")

        image_files = sorted(self.img_dir.glob("*.png"))
        self.files = [p for p in image_files if (self.ann_dir / p.name).is_file()]
        if max_items is not None:
            self.files = self.files[:max_items]

        if not self.files:
            raise RuntimeError(
                "No matched PNG image/annotation pairs found under "
                f"img_dir={self.img_dir} and ann_dir={self.ann_dir}"
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        ann_path = self.ann_dir / img_path.name

        raw_gt = Image.open(ann_path)
        raw_gt = raw_gt.resize((self.resize, self.resize), Image.NEAREST)
        raw_gt = np.array(raw_gt)
        gt = remap_isprs_raw_gt(raw_gt)

        return {
            "img_path": str(img_path),
            "gt": torch.from_numpy(gt).long(),
            "name": img_path.stem,
        }


class DynamicLoRALinear(nn.Module):

    def __init__(
        self,
        base: nn.Linear,
        max_r: int,
        alpha_mult: float = 2.0,
        dropout: float = 0.0,
        block_id: Optional[int] = None,
        init_active_r: Optional[int] = None,
    ):
        super().__init__()
        assert isinstance(base, nn.Linear)

        self.base = base
        self.block_id = -1 if block_id is None else int(block_id)

        self.in_features = base.in_features
        self.out_features = base.out_features
        self.max_r = int(max_r)
        self.alpha_mult = float(alpha_mult)

        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        dev = self.base.weight.device
        dt = self.base.weight.dtype

        self.lora_A = nn.Parameter(torch.zeros(self.max_r, self.in_features, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.max_r, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if init_active_r is None:
            init_active_r = self.max_r
        init_active_r = int(max(1, min(init_active_r, self.max_r)))
        self.register_buffer("active_r", torch.tensor(init_active_r, dtype=torch.int32), persistent=False)
        self.register_buffer("global_scale", torch.tensor(1.0, dtype=torch.float32), persistent=False)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def set_active_rank(self, r: int):
        r = int(max(1, min(r, self.max_r)))
        self.active_r.fill_(r)

    def get_active_rank(self) -> int:
        return int(self.active_r.item())

    def set_global_scale(self, scale: float):
        self.global_scale.fill_(float(scale))

    def forward(self, x):
        out = self.base(x)
        r = self.get_active_rank()
        if r > 0:
            A = self.lora_A[:r].to(device=x.device, dtype=x.dtype)
            B = self.lora_B[:, :r].to(device=x.device, dtype=x.dtype)
            lora_out = F.linear(self.dropout(x), A)
            lora_out = F.linear(lora_out, B)
            out = out + self.global_scale.to(device=x.device, dtype=x.dtype) * self.alpha_mult * lora_out
        return out


def _get_parent_module(root: nn.Module, module_name: str):
    parts = module_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _extract_block_idx(name: str):
    m = re.search(r"\bblocks\.(\d+)\b", name)
    if m:
        return int(m.group(1))
    m = re.search(r"\blayers\.(\d+)\b", name)
    if m:
        return int(m.group(1))
    return None


def find_lora_targets(
    model: nn.Module,
    block_indices: List[int],
    attn_keywords: Tuple[str, ...] = ("attn",),
    proj_keywords: Tuple[str, ...] = ("q", "k", "v", "proj", "out", "fc"),
    backbone_only: bool = True,
):
    linear_names = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        lname = name.lower()

        if backbone_only and "backbone.vision_backbone.trunk.blocks." not in lname:
            continue

        if not any(k in lname for k in attn_keywords):
            continue

        idx = _extract_block_idx(lname)
        linear_names.append((name, idx))

    keep_blocks = set(int(x) for x in block_indices)
    targets = []
    for name, idx in linear_names:
        lname = name.lower()
        if idx in keep_blocks and any(k in lname for k in proj_keywords):
            targets.append(name)

    return sorted(set(targets))


def inject_dynamic_lora(
    model: nn.Module,
    target_names: List[str],
    max_r: int,
    init_active_r: int,
    alpha_mult: float,
    dropout: float,
):
    replaced = []
    for name in target_names:
        parent, child_name = _get_parent_module(model, name)
        child = getattr(parent, child_name)
        if not isinstance(child, nn.Linear):
            continue

        block_id = _extract_block_idx(name.lower())
        new_mod = DynamicLoRALinear(
            base=child,
            max_r=max_r,
            init_active_r=init_active_r,
            alpha_mult=alpha_mult,
            dropout=dropout,
            block_id=block_id,
        ).to(child.weight.device)

        setattr(parent, child_name, new_mod)
        replaced.append(name)

    return replaced


def mark_only_lora_trainable(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False

    trainable = []
    for n, p in model.named_parameters():
        if ("lora_A" in n) or ("lora_B" in n):
            p.requires_grad = True
            trainable.append(n)
    return trainable


def get_trainable_state_dict(model: nn.Module):
    return {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if ("lora_A" in k or "lora_B" in k)
    }


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def set_global_lora_scale(model: nn.Module, scale: float):
    for m in model.modules():
        if isinstance(m, DynamicLoRALinear):
            m.set_global_scale(scale)


def set_block_active_ranks(model: nn.Module, block_rank_map: Dict[int, int]):
    for m in model.modules():
        if isinstance(m, DynamicLoRALinear):
            bid = int(m.block_id)
            if bid in block_rank_map:
                m.set_active_rank(int(block_rank_map[bid]))


def _low_rank_frobenius_sq(B: torch.Tensor, A: torch.Tensor) -> float:

    gram_b = B.transpose(0, 1) @ B
    gram_a = A @ A.transpose(0, 1)
    value = (gram_b * gram_a.transpose(0, 1)).sum()
    return float(torch.clamp(value, min=0.0).item())


def _low_rank_relative_error(
    B_ref: torch.Tensor,
    A_ref: torch.Tensor,
    B_new: torch.Tensor,
    A_new: torch.Tensor,
) -> Tuple[float, float, float]:

    ref_sq = _low_rank_frobenius_sq(B_ref, A_ref)
    new_sq = _low_rank_frobenius_sq(B_new, A_new)

    cross_b = B_ref.transpose(0, 1) @ B_new
    cross_a = A_new @ A_ref.transpose(0, 1)
    cross = float((cross_b * cross_a.transpose(0, 1)).sum().item())
    diff_sq = max(ref_sq + new_sq - 2.0 * cross, 0.0)

    if ref_sq <= 1e-30:
        rel_error = 0.0 if diff_sq <= 1e-30 else float("inf")
    else:
        rel_error = math.sqrt(diff_sq / ref_sq)
    return rel_error, ref_sq, diff_sq


@torch.no_grad()
def svd_preserve_lora_updates(
    model: nn.Module,
    block_rank_map: Dict[int, int],
) -> Dict[str, object]:

    locked_map = normalize_rank_map(block_rank_map)
    lora_modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, DynamicLoRALinear)
        and int(module.block_id) in locked_map
    ]
    if not lora_modules:
        raise RuntimeError("No DynamicLoRALinear modules found for SVD conversion.")

    module_reports = []
    block_totals: Dict[int, Dict[str, float]] = {}
    global_ref_sq = 0.0
    global_diff_sq = 0.0

    for index, (name, module) in enumerate(lora_modules, start=1):
        old_rank = int(module.get_active_rank())
        new_rank = int(locked_map[int(module.block_id)])
        if not 1 <= new_rank <= module.max_r:
            raise ValueError(
                f"Invalid allocated rank for {name}: {new_rank} (max={module.max_r})"
            )

        device = module.lora_A.device
        old_A = module.lora_A[:old_rank].detach().float().clone()
        old_B = module.lora_B[:, :old_rank].detach().float().clone()
        extra_A = None
        if new_rank > old_rank:
            extra_A = module.lora_A[old_rank:new_rank].detach().clone()


        q_b, r_b = torch.linalg.qr(old_B, mode="reduced")
        q_a, r_a = torch.linalg.qr(old_A.transpose(0, 1), mode="reduced")
        core = r_b @ r_a.transpose(0, 1)
        try:
            u_core, singular_values, vh_core = torch.linalg.svd(
                core,
                full_matrices=False,
            )
        except RuntimeError as error:
            print(
                f"[SVD rank conversion] CUDA/device SVD failed for {name}; "
                f"retrying the {tuple(core.shape)} core on CPU: {error}",
                flush=True,
            )
            u_cpu, s_cpu, vh_cpu = torch.linalg.svd(
                core.cpu(),
                full_matrices=False,
            )
            u_core = u_cpu.to(device=device)
            singular_values = s_cpu.to(device=device)
            vh_core = vh_cpu.to(device=device)

        kept_rank = min(
            new_rank,
            old_rank,
            int(singular_values.numel()),
        )
        sqrt_s = singular_values[:kept_rank].clamp_min(0.0).sqrt()
        new_B_factor = (
            q_b @ u_core[:, :kept_rank]
        ) * sqrt_s.unsqueeze(0)
        new_A_factor = sqrt_s.unsqueeze(1) * (
            vh_core[:kept_rank] @ q_a.transpose(0, 1)
        )


        module.lora_A[:new_rank].zero_()
        module.lora_B[:, :new_rank].zero_()
        module.lora_A[:kept_rank].copy_(
            new_A_factor.to(dtype=module.lora_A.dtype)
        )
        module.lora_B[:, :kept_rank].copy_(
            new_B_factor.to(dtype=module.lora_B.dtype)
        )
        if extra_A is not None:
            module.lora_A[old_rank:new_rank].copy_(extra_A)
        module.set_active_rank(new_rank)


        stored_A = module.lora_A[:new_rank].detach().float()
        stored_B = module.lora_B[:, :new_rank].detach().float()
        rel_error, ref_sq, diff_sq = _low_rank_relative_error(
            old_B,
            old_A,
            stored_B,
            stored_A,
        )
        retained_energy = (
            max(0.0, 1.0 - diff_sq / ref_sq)
            if ref_sq > 1e-30
            else 1.0
        )

        report = {
            "module": name,
            "block": int(module.block_id),
            "probe_rank": old_rank,
            "allocated_rank": new_rank,
            "svd_kept_rank": kept_rank,
            "reference_frobenius_sq": ref_sq,
            "reconstruction_diff_sq": diff_sq,
            "relative_frobenius_error": rel_error,
            "retained_energy": retained_energy,
        }
        module_reports.append(report)
        global_ref_sq += ref_sq
        global_diff_sq += diff_sq

        block_stat = block_totals.setdefault(
            int(module.block_id),
            {"reference_frobenius_sq": 0.0, "reconstruction_diff_sq": 0.0},
        )
        block_stat["reference_frobenius_sq"] += ref_sq
        block_stat["reconstruction_diff_sq"] += diff_sq

    block_reports = {}
    for block_id, stat in sorted(block_totals.items()):
        ref_sq = stat["reference_frobenius_sq"]
        diff_sq = stat["reconstruction_diff_sq"]
        block_reports[str(block_id)] = {
            **stat,
            "relative_frobenius_error": (
                math.sqrt(max(diff_sq, 0.0) / ref_sq)
                if ref_sq > 1e-30
                else 0.0
            ),
            "retained_energy": (
                max(0.0, 1.0 - diff_sq / ref_sq)
                if ref_sq > 1e-30
                else 1.0
            ),
        }

    global_relative_error = (
        math.sqrt(max(global_diff_sq, 0.0) / global_ref_sq)
        if global_ref_sq > 1e-30
        else 0.0
    )
    return {
        "method": "compact_truncated_svd",
        "module_count": len(module_reports),
        "global_reference_frobenius_sq": global_ref_sq,
        "global_reconstruction_diff_sq": global_diff_sq,
        "global_relative_frobenius_error": global_relative_error,
        "global_retained_energy": (
            max(0.0, 1.0 - global_diff_sq / global_ref_sq)
            if global_ref_sq > 1e-30
            else 1.0
        ),
        "blocks": block_reports,
        "modules": module_reports,
    }


def collect_block_active_ranks(model: nn.Module):
    ranks = {}
    for _, m in model.named_modules():
        if isinstance(m, DynamicLoRALinear):
            ranks.setdefault(int(m.block_id), []).append(m.get_active_rank())
    return {k: int(round(sum(v) / len(v))) for k, v in sorted(ranks.items(), key=lambda x: x[0])}


def build_semantic_logits_for_image(processor, image_pil, device):
    H, W = image_pil.size[1], image_pil.size[0]

    state = processor.set_image(image_pil)
    fg_logits = []

    for prompt in PROMPT_CLASS_NAMES:
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(state=state, prompt=prompt)

        sem = state.get("semantic_mask_logits", None)

        if sem is None:
            sem = torch.full((1, 1, H, W), -1e4, device=device, dtype=torch.float32)
        else:
            if sem.ndim == 3:
                sem = sem.unsqueeze(1)
            sem = sem.float()
            if sem.shape[-2:] != (H, W):
                sem = F.interpolate(sem, size=(H, W), mode="bilinear", align_corners=False)

        fg_logits.append(sem)

    fg_logits = torch.cat(fg_logits, dim=1)
    bg_logit = -fg_logits.max(dim=1, keepdim=True).values


    logits = torch.cat([fg_logits, bg_logit], dim=1)
    if logits.shape[1] != NUM_CLASSES:
        raise RuntimeError(
            f"Semantic logits have {logits.shape[1]} channels, "
            f"but NUM_CLASSES={NUM_CLASSES}. "
            f"PROMPT_CLASS_NAMES={PROMPT_CLASS_NAMES}"
        )
    return logits


def unwrap_processor_for_training(processor):
    if hasattr(processor.set_image, "__wrapped__"):
        processor.set_image = processor.set_image.__wrapped__.__get__(processor, processor.__class__)
    if hasattr(processor.set_text_prompt, "__wrapped__"):
        processor.set_text_prompt = processor.set_text_prompt.__wrapped__.__get__(processor, processor.__class__)
    if hasattr(processor._forward_grounding, "__wrapped__"):
        processor._forward_grounding = processor._forward_grounding.__wrapped__.__get__(processor, processor.__class__)


def train_one_epoch(processor, loader, optimizer, device, resize, class_weights=None, desc="Training"):
    processor.model.eval()
    running = 0.0

    if class_weights is not None:
        class_weights = class_weights.to(device)

    for batch in tqdm(loader, desc=desc, leave=False):
        img_path = batch["img_path"][0]
        image = Image.open(img_path).convert("RGB")
        image = image.resize((resize, resize), Image.BILINEAR)

        gt = batch["gt"][0].to(device)

        logits = build_semantic_logits_for_image(processor, image, device)

        if logits.shape[-2:] != gt.shape[-2:]:
            logits = F.interpolate(logits, size=gt.shape[-2:], mode="bilinear", align_corners=False)

        loss = F.cross_entropy(
            logits,
            gt.unsqueeze(0),
            ignore_index=IGNORE_INDEX,
            weight=class_weights,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running += float(loss.item())

    return running / max(len(loader), 1)


@torch.no_grad()
def evaluate(processor, loader, device, resize, desc="Evaluating"):
    total_inter = np.zeros(NUM_CLASSES, dtype=np.float64)
    total_union = np.zeros(NUM_CLASSES, dtype=np.float64)

    processor.model.eval()

    for batch in tqdm(loader, desc=desc, leave=False):
        img_path = batch["img_path"][0]
        image = Image.open(img_path).convert("RGB")
        image = image.resize((resize, resize), Image.BILINEAR)

        gt = batch["gt"][0].numpy()

        logits = build_semantic_logits_for_image(processor, image, device)

        if logits.shape[-2:] != gt.shape[-2:]:
            logits = F.interpolate(logits, size=gt.shape[-2:], mode="bilinear", align_corners=False)

        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        valid = (gt != IGNORE_INDEX)
        pred = pred[valid]
        gt = gt[valid]

        for c in range(NUM_CLASSES):
            inter = np.logical_and(pred == c, gt == c).sum()
            union = np.logical_or(pred == c, gt == c).sum()
            total_inter[c] += inter
            total_union[c] += union

    ious = total_inter / np.maximum(total_union, 1.0)
    miou = float(np.mean(ious))
    return miou, ious.tolist()


@torch.no_grad()
def collect_block_scores(model: nn.Module, score_mode: str = "fro_sq") -> Dict[int, float]:

    scores = {}

    for _, m in model.named_modules():
        if not isinstance(m, DynamicLoRALinear):
            continue

        r = m.get_active_rank()
        if r <= 0:
            continue

        A = m.lora_A[:r].detach().float().cpu()
        B = m.lora_B[:, :r].detach().float().cpu()
        delta = B @ A

        if score_mode == "fro_sq":
            score = float((delta ** 2).sum().item())
        else:
            sv = torch.linalg.svdvals(delta)
            if score_mode == "sv_sum":
                score = float(sv.sum().item())
            elif score_mode == "sv_sq_sum":
                score = float((sv ** 2).sum().item())
            else:
                raise ValueError(f"Unknown score_mode: {score_mode}")

        bid = int(m.block_id)
        scores.setdefault(bid, []).append(score)

    scores = {k: float(sum(v) / max(len(v), 1)) for k, v in scores.items()}
    return dict(sorted(scores.items(), key=lambda x: x[0]))


def allocate_block_ranks(
    candidate_blocks: List[int],
    scores: Dict[int, float],
    total_rank_budget: int,
    r_min: int,
    r_max: int,
) -> Dict[int, int]:
    candidate_blocks = [int(x) for x in candidate_blocks]
    n = len(candidate_blocks)
    assert n > 0
    assert total_rank_budget >= n * r_min, "total_rank_budget too small"

    raw_scores = np.array([float(scores.get(b, 0.0)) for b in candidate_blocks], dtype=np.float64)
    if np.all(raw_scores <= 0):
        raw_scores = np.ones_like(raw_scores)

    budget_left = total_rank_budget - n * r_min
    base = np.full(n, r_min, dtype=np.int64)

    if budget_left > 0:
        probs = raw_scores / np.maximum(raw_scores.sum(), 1e-12)
        extra_float = probs * budget_left
        extra_int = np.floor(extra_float).astype(np.int64)
        base += extra_int

        remainder = int(budget_left - extra_int.sum())
        frac = extra_float - extra_int
        order = np.argsort(-frac)
        for i in range(remainder):
            base[order[i]] += 1

    base = np.minimum(base, r_max)

    current = int(base.sum())
    while current < total_rank_budget:
        candidates = [i for i in range(n) if base[i] < r_max]
        if len(candidates) == 0:
            break
        best_i = max(candidates, key=lambda i: raw_scores[i])
        base[best_i] += 1
        current += 1

    while current > total_rank_budget:
        candidates = [i for i in range(n) if base[i] > r_min]
        if len(candidates) == 0:
            break
        worst_i = min(candidates, key=lambda i: raw_scores[i])
        base[worst_i] -= 1
        current -= 1

    return {b: int(r) for b, r in zip(candidate_blocks, base.tolist())}


def normalize_rank_map(rank_map: Dict[int, int]) -> Dict[int, int]:
    return {
        int(k): int(v)
        for k, v in sorted(rank_map.items(), key=lambda item: int(item[0]))
    }


def get_global_lora_scale(epoch: int, total_epochs: int, min_scale: float = 0.8) -> float:
    progress = (epoch - 1) / max(total_epochs - 1, 1)
    cos_part = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_scale + (1.0 - min_scale) * cos_part)


def build_model_with_dynamic_lora(
    checkpoint: str,
    device: torch.device,
    candidate_blocks: List[int],
    rank_max: int,
    init_active_r: int,
    alpha_mult: float,
    dropout: float,
):
    model = build_sam3_image_model(
        checkpoint_path=checkpoint,
        device=str(device),
        eval_mode=True,
    )

    targets = find_lora_targets(
        model,
        block_indices=candidate_blocks,
        backbone_only=True,
    )

    injected = inject_dynamic_lora(
        model,
        target_names=targets,
        max_r=rank_max,
        init_active_r=init_active_r,
        alpha_mult=alpha_mult,
        dropout=dropout,
    )

    model.to(device)
    model.eval()

    trainable_names = mark_only_lora_trainable(model)
    total_params, trainable_params = count_parameters(model)

    processor = Sam3Processor(model=model, device=str(device))
    unwrap_processor_for_training(processor)

    return model, processor, targets, injected, trainable_names, total_params, trainable_params


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, default="checkpoints/pretrained/sam3.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resize", type=int, default=512)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument(
        "--train-img-dir",
        type=str,
        required=True,
        help="Source-domain training image directory (PNG files).",
    )
    parser.add_argument(
        "--train-ann-dir",
        type=str,
        required=True,
        help="Source-domain training annotation directory (same PNG filenames).",
    )


    parser.add_argument("--probe-r", type=int, default=128)


    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-train", type=int, default=None)

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--alpha-multiplier", type=float, default=2.0)

    parser.add_argument(
        "--candidate-blocks",
        type=str,
        default="3,7,11,15,19,23,27,31",
        help="Comma-separated candidate SAM3 trunk block indices"
    )

    parser.add_argument("--total-rank-budget", type=int, default=512)
    parser.add_argument("--rank-min", type=int, default=4)
    parser.add_argument("--rank-max", type=int, default=256)
    parser.add_argument("--score-mode", type=str, default="fro_sq", choices=["fro_sq", "sv_sum", "sv_sq_sum"])
    parser.add_argument(
        "--global-scale-min",
        type=float,
        default=0.8,
        help="Minimum training-time LoRA scale for the cosine schedule.",
    )
    parser.add_argument(
        "--eval-global-scale",
        type=float,
        default=0.8,
        help=(
            "Fixed LoRA scale used for every final-stage target validation. "
            "The final weights use this inference scale."
        ),
    )

    parser.add_argument(
        "--class-weights",
        type=str,
        default="1,1,1,1,1,1",
        help=(
            "impervious_surface, building, low_vegetation, tree, car, "
            "clutter_background"
        ),
    )

    parser.add_argument(
        "--work-dir",
        type=str,
        default=(
            "checkpoints/"
            "isprs_pi2vi/tcsb"
        ),
    )


    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    blocks = [int(value) for value in args.candidate_blocks.split(",")]
    class_weights = torch.tensor([float(v) for v in args.class_weights.split(",")], dtype=torch.float32)
    train_set = ISPRSSingleImageDataset(args.train_img_dir, args.train_ann_dir, args.resize, args.max_train)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
    model, processor, targets, injected, names, total, trainable = build_model_with_dynamic_lora(
        checkpoint=args.checkpoint, device=device, candidate_blocks=blocks,
        rank_max=args.rank_max, init_active_r=args.probe_r,
        alpha_mult=args.alpha_multiplier, dropout=args.lora_dropout,
    )
    def make_optimizer():
        return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=args.lr, weight_decay=args.weight_decay)
    optimizer = make_optimizer()

    set_global_lora_scale(model, 1.0)
    probe_loss = train_one_epoch(processor, train_loader, optimizer, device,
                                 args.resize, class_weights, desc="Probe")
    print(f"[TCSB] Probe loss={probe_loss:.4f}", flush=True)
    scores = collect_block_scores(model, score_mode=args.score_mode)
    rank_map = normalize_rank_map(allocate_block_ranks(
        candidate_blocks=blocks, scores=scores, total_rank_budget=args.total_rank_budget,
        r_min=args.rank_min, r_max=args.rank_max,
    ))
    print(f"[TCSB] Rank allocation | total={sum(rank_map.values())} "
          f"ranks={dict(sorted(rank_map.items()))}", flush=True)
    print("[TCSB] Converting LoRA ranks (SVD)", flush=True)
    svd_preserve_lora_updates(model, rank_map)

    optimizer = make_optimizer()
    history = []
    for epoch in range(1, args.epochs + 1):
        scale = get_global_lora_scale(epoch, args.epochs, min_scale=args.global_scale_min)
        set_global_lora_scale(model, scale)
        loss = train_one_epoch(processor, train_loader, optimizer, device,
                               args.resize, class_weights, desc=f"TCSB {epoch}/{args.epochs}")
        history.append({"epoch": epoch, "loss": float(loss)})
        print(f"[TCSB] Epoch {epoch}/{args.epochs} | loss={loss:.4f}", flush=True)
    set_global_lora_scale(model, float(args.eval_global_scale))
    torch.save({
        "epoch": args.epochs,
        "lora_state_dict": get_trainable_state_dict(model),
        "candidate_blocks": blocks,
        "final_rank_map": rank_map,
        "global_lora_scale": float(args.eval_global_scale),
        "eval_global_lora_scale": float(args.eval_global_scale),
        "class_names": CLASS_NAMES,
        "args": vars(args),
    }, work_dir / "final_lora.pth")
    with (work_dir / "history.json").open("w") as handle:
        json.dump({"probe_loss": float(probe_loss), "training": history}, handle, indent=2)
    print(f"[TCSB] Saved: {work_dir / 'final_lora.pth'}", flush=True)


if __name__ == "__main__":
    main()
