from __future__ import annotations

from method.logging_utils import log_epoch, log_fusion_evaluation

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader
from method.logging_utils import progress as tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from method import tsdb_loveda as loveda_b3
from method import tcsb_loveda as loveda_base


CLASS_NAMES = list(loveda_base.CLASS_NAMES)
NUM_CLASSES = int(loveda_base.NUM_CLASSES)
IGNORE_INDEX = int(loveda_base.IGNORE_INDEX)


def _safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _rgb01(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _rgb_pil(x: np.ndarray) -> Image.Image:
    return Image.fromarray(
        np.round(np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB"
    )


def _smooth_field(height: int, width: int, grid: int = 6) -> np.ndarray:
    small = np.random.uniform(-1.0, 1.0, size=(grid, grid)).astype(np.float32)
    field = Image.fromarray(
        np.round((small + 1.0) * 127.5).astype(np.uint8), mode="L"
    )
    field = field.resize((width, height), Image.BICUBIC)
    return np.asarray(field, dtype=np.float32) / 127.5 - 1.0


def _spectral_shift(x: np.ndarray, strength: float) -> np.ndarray:
    s = float(strength)
    if random.random() < 0.45:
        red, green, blue = x[..., 0], x[..., 1], x[..., 2]
        green_excess = np.maximum(2.0 * green - red - blue, 0.0)
        pseudo_nir = np.clip(
            0.30 * red
            + 0.55 * green
            + 0.15 * blue
            + 0.30 * green_excess,
            0.0,
            1.0,
        )
        pseudo_irrg = np.stack([pseudo_nir, red, green], axis=-1)
        amount = random.uniform(0.45, 0.80) * s
        x = (1.0 - amount) * x + amount * pseudo_irrg
    else:
        response_random = np.random.gamma(1.5, 1.0, size=(3, 3)).astype(
            np.float32
        )
        response_random /= np.maximum(
            response_random.sum(axis=1, keepdims=True), 1e-6
        )
        amount = random.uniform(0.30, 0.65) * s
        response = (
            (1.0 - amount) * np.eye(3, dtype=np.float32)
            + amount * response_random
        )
        x = np.einsum("hwc,dc->hwd", x, response)

    gain = np.random.uniform(
        1.0 - 0.30 * s, 1.0 + 0.30 * s, (1, 1, 3)
    ).astype(np.float32)
    bias = np.random.uniform(-0.08 * s, 0.08 * s, (1, 1, 3)).astype(
        np.float32
    )
    gamma = np.random.uniform(
        max(0.55, 1.0 - 0.38 * s), 1.0 + 0.45 * s, (1, 1, 3)
    ).astype(np.float32)
    return np.power(np.clip(x * gain + bias, 0.0, 1.0), gamma)


def _weather_shift(x: np.ndarray, strength: float) -> np.ndarray:
    s = float(strength)
    height, width, _ = x.shape
    field = _smooth_field(height, width, grid=random.choice((4, 5, 6)))
    if random.random() < 0.60:
        base_haze = random.uniform(0.06, 0.28) * s
        transmission = np.clip(
            1.0 - base_haze * (1.0 + 0.45 * field), 0.55, 1.0
        )
        airlight = np.random.uniform(0.82, 1.0, size=(1, 1, 3)).astype(
            np.float32
        )
        x = (
            x * transmission[..., None]
            + airlight * (1.0 - transmission[..., None])
        )
    illumination = 1.0 + random.uniform(0.08, 0.28) * s * field[..., None]
    temperature = np.array(
        [
            random.uniform(1.0 - 0.15 * s, 1.0 + 0.18 * s),
            random.uniform(1.0 - 0.06 * s, 1.0 + 0.06 * s),
            random.uniform(1.0 - 0.18 * s, 1.0 + 0.15 * s),
        ],
        dtype=np.float32,
    ).reshape(1, 1, 3)
    return x * illumination * temperature


def _sensor_shift(x: np.ndarray, strength: float) -> np.ndarray:
    s = float(strength)
    height, width, _ = x.shape
    image = _rgb_pil(x)
    factor = 1.0 + random.uniform(0.30, 1.40) * s
    small = (max(8, round(width / factor)), max(8, round(height / factor)))
    image = image.resize(small, Image.BILINEAR).resize(
        (width, height), Image.BILINEAR
    )
    if random.random() < 0.65:
        image = image.filter(
            ImageFilter.GaussianBlur(radius=random.uniform(0.15, 0.85) * s)
        )
    x = _rgb01(image)
    sigma = random.uniform(0.003, 0.018) * s
    noise_scale = sigma * np.sqrt(np.clip(x, 0.05, 1.0)) + 0.25 * sigma
    return x + np.random.normal(0.0, 1.0, size=x.shape).astype(
        np.float32
    ) * noise_scale


def _frequency_shift(x: np.ndarray, strength: float) -> np.ndarray:
    s = float(strength)
    height, width, _ = x.shape
    yy, xx = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2
    radius = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    low_cut = min(height, width) * random.uniform(0.035, 0.10)
    high_cut = min(height, width) * random.uniform(0.22, 0.38)
    low_mask = radius <= low_cut
    high_mask = radius >= high_cut
    low_gain = random.uniform(1.0 - 0.55 * s, 1.0 + 0.55 * s)
    high_gain = random.uniform(1.0 - 0.65 * s, 1.0 + 0.30 * s)
    output = np.empty_like(x)
    for channel in range(3):
        spectrum = np.fft.fftshift(np.fft.fft2(x[..., channel]))
        amplitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        dc_value = amplitude[center_y, center_x].copy()
        amplitude[low_mask] *= low_gain * random.uniform(0.94, 1.06)
        amplitude[high_mask] *= high_gain
        amplitude[center_y, center_x] = dc_value
        restored = amplitude * np.exp(1j * phase)
        output[..., channel] = np.real(
            np.fft.ifft2(np.fft.ifftshift(restored))
        )
    return output


def domain_randomize(
    image: Image.Image,
    strength: float = 0.75,
) -> Tuple[Image.Image, Dict[str, float], str]:

    scaled_strength = float(np.clip(strength, 0.0, 1.0))
    clean = _rgb01(image)
    output = clean.copy()
    operations = {
        "spectral": _spectral_shift,
        "weather": _weather_shift,
        "sensor": _sensor_shift,
        "frequency": _frequency_shift,
    }
    names = list(operations)
    primary = random.choices(
        names, weights=(0.38, 0.24, 0.20, 0.18), k=1
    )[0]
    output = operations[primary](output, scaled_strength)
    used = [primary]
    if random.random() < 0.40:
        secondary = random.choice([name for name in names if name != primary])
        output = operations[secondary](
            np.clip(output, 0.0, 1.0), 0.40 * scaled_strength
        )
        used.append(secondary)
    output = np.clip(output, 0.0, 1.0)
    delta = output - clean
    stats = {
        "aug_mad": float(np.abs(delta).mean()),
        "aug_rms": float(np.sqrt(np.square(delta).mean())),
    }
    return _rgb_pil(output), stats, "+".join(used)


def _normalized_entropy(probability: torch.Tensor) -> torch.Tensor:
    entropy = -(probability.clamp_min(1e-8).log() * probability).sum(
        dim=1, keepdim=True
    )
    return entropy / math.log(probability.shape[1])


def _top1_margin(probability: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    top2 = probability.topk(k=2, dim=1).values
    confidence = top2[:, :1]
    margin = top2[:, :1] - top2[:, 1:2]
    return confidence, margin


def build_router_features(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
) -> torch.Tensor:

    probability_a = F.softmax(logits_a.float(), dim=1)
    probability_b = F.softmax(logits_b.float(), dim=1)
    prediction_a = probability_a.argmax(dim=1)
    prediction_b = probability_b.argmax(dim=1)

    one_hot_a = F.one_hot(prediction_a, NUM_CLASSES).permute(0, 3, 1, 2).float()
    one_hot_b = F.one_hot(prediction_b, NUM_CLASSES).permute(0, 3, 1, 2).float()
    confidence_a, margin_a = _top1_margin(probability_a)
    confidence_b, margin_b = _top1_margin(probability_b)
    disagreement = (prediction_a != prediction_b).unsqueeze(1).float()


    return torch.cat(
        [
            probability_a,
            probability_b,
            (probability_a - probability_b).abs(),
            one_hot_a,
            one_hot_b,
            _normalized_entropy(probability_a),
            _normalized_entropy(probability_b),
            confidence_a,
            confidence_b,
            margin_a,
            margin_b,
            disagreement,
        ],
        dim=1,
    )


class ResidualDepthwiseBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        groups = 8 if channels % 8 == 0 else 4
        self.norm = nn.GroupNorm(groups, channels)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.activation(x + residual)


class SelectiveRescueRouter(nn.Module):


    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        hidden_dim: int = 32,
        router_scale: float = 0.5,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.in_channels = 5 * self.num_classes + 7
        self.hidden_dim = int(hidden_dim)
        self.router_scale = float(router_scale)
        if not 0.0 < self.router_scale <= 1.0:
            raise ValueError("router_scale must be in (0, 1]")

        groups = 8 if self.hidden_dim % 8 == 0 else 4
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            ResidualDepthwiseBlock(self.hidden_dim, dilation=1),
            ResidualDepthwiseBlock(self.hidden_dim, dilation=2),
        )
        self.output = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)


        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.output.bias, -2.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output_size = features.shape[-2:]
        if self.router_scale < 1.0:
            scaled_size = (
                max(1, int(round(output_size[0] * self.router_scale))),
                max(1, int(round(output_size[1] * self.router_scale))),
            )
            features = F.interpolate(
                features,
                size=scaled_size,
                mode="bilinear",
                align_corners=False,
            )
        x = self.stem(features)
        x = self.context(x)
        route_logits = self.output(x)
        if route_logits.shape[-2:] != output_size:
            route_logits = F.interpolate(
                route_logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        return route_logits


def relation_masks(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    if target.ndim == 2:
        target = target.unsqueeze(0)
    if target.ndim != 3:
        raise ValueError(
            f"Expected target with shape [H, W] or [B, H, W], got {tuple(target.shape)}"
        )
    prediction_a = logits_a.argmax(dim=1)
    prediction_b = logits_b.argmax(dim=1)
    if target.shape != prediction_a.shape or target.shape != prediction_b.shape:
        raise ValueError(
            "Target/expert prediction shape mismatch: "
            f"target={tuple(target.shape)}, A={tuple(prediction_a.shape)}, "
            f"B={tuple(prediction_b.shape)}"
        )
    valid = target != IGNORE_INDEX
    a_correct = prediction_a == target
    b_correct = prediction_b == target
    disagreement = prediction_a != prediction_b
    positive = valid & disagreement & (~a_correct) & b_correct
    negative = valid & disagreement & a_correct & (~b_correct)
    both_wrong = valid & disagreement & (~a_correct) & (~b_correct)
    supervised = positive | negative
    return {


        "target": target,
        "valid": valid,
        "disagreement": valid & disagreement,
        "positive": positive,
        "negative": negative,
        "both_wrong": both_wrong,
        "supervised": supervised,
        "prediction_a": prediction_a,
        "prediction_b": prediction_b,
    }


def bounded_soft_fusion(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    route_logits: torch.Tensor,
    max_b_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    probability_a = F.softmax(logits_a.float(), dim=1)
    probability_b = F.softmax(logits_b.float(), dim=1)
    gate = route_logits.sigmoid()
    weight_b = float(max_b_weight) * gate
    probability_fused = (
        (1.0 - weight_b) * probability_a + weight_b * probability_b
    )
    probability_fused = probability_fused / probability_fused.sum(
        dim=1, keepdim=True
    ).clamp_min(1e-8)
    return probability_fused, weight_b, gate


def fused_cross_entropy(
    probability_fused: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if target.ndim == 2:
        target = target.unsqueeze(0)
    return F.nll_loss(
        probability_fused.clamp_min(1e-8).log(),
        target.long(),
        ignore_index=IGNORE_INDEX,
    )


def selective_xded_loss(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    weight_b: torch.Tensor,
    masks: Dict[str, torch.Tensor],
    temperature: float,
    negative_cost: float,
) -> torch.Tensor:

    supervised = masks["supervised"].unsqueeze(1)
    if int(supervised.sum().item()) == 0:
        return weight_b.sum() * 0.0

    temperature = float(temperature)
    probability_a_t = F.softmax(logits_a.float() / temperature, dim=1)
    probability_b_t = F.softmax(logits_b.float() / temperature, dim=1)
    probability_fused_t = (
        (1.0 - weight_b) * probability_a_t + weight_b * probability_b_t
    ).clamp_min(1e-8)

    teacher = torch.where(
        masks["positive"].unsqueeze(1),
        probability_b_t,
        probability_a_t,
    ).detach().clamp_min(1e-8)
    per_pixel_kl = (
        teacher * (teacher.log() - probability_fused_t.log())
    ).sum(dim=1, keepdim=True) * (temperature ** 2)

    pixel_weight = supervised.float()
    pixel_weight = torch.where(
        masks["negative"].unsqueeze(1),
        pixel_weight.new_full((), float(negative_cost)),
        pixel_weight,
    ) * supervised.float()
    return (per_pixel_kl * pixel_weight).sum() / pixel_weight.sum().clamp_min(1.0)


def router_consistency_loss(
    gate_original: torch.Tensor,
    gate_augmented: torch.Tensor,
    disagreement_original: torch.Tensor,
    disagreement_augmented: torch.Tensor,
) -> torch.Tensor:

    mask = (disagreement_original | disagreement_augmented).unsqueeze(1)
    if int(mask.sum().item()) == 0:
        return gate_original.sum() * 0.0
    per_pixel = F.smooth_l1_loss(
        gate_original,
        gate_augmented,
        reduction="none",
    )
    mask_float = mask.float()
    return (per_pixel * mask_float).sum() / mask_float.sum().clamp_min(1.0)


def _update_confusion(
    confusion: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
) -> None:
    encoded = target.astype(np.int64) * NUM_CLASSES + prediction.astype(np.int64)
    confusion += np.bincount(
        encoded,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(NUM_CLASSES, NUM_CLASSES)


def _scores(confusion: np.ndarray) -> Tuple[float, Sequence[float], float]:
    true_positive = np.diag(confusion).astype(np.float64)
    union = confusion.sum(axis=1) + confusion.sum(axis=0) - true_positive
    iou = true_positive / np.maximum(union, 1.0)
    pixel_accuracy = true_positive.sum() / max(float(confusion.sum()), 1.0)
    return float(iou.mean()), iou.tolist(), float(pixel_accuracy)


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(float(denominator), 1.0)


def _accumulate_fusion_stats(
    totals: Dict[str, float],
    masks: Dict[str, torch.Tensor],
    fused_prediction: torch.Tensor,
    gate: torch.Tensor,
    weight_b: torch.Tensor,
) -> None:
    positive = masks["positive"]
    negative = masks["negative"]
    supervised = masks["supervised"]
    disagreement = masks["disagreement"]
    valid = masks["valid"]

    totals["valid"] += float(valid.sum().item())
    totals["disagreement"] += float(disagreement.sum().item())
    totals["positive"] += float(positive.sum().item())
    totals["negative"] += float(negative.sum().item())
    totals["both_wrong_disagreement"] += float(masks["both_wrong"].sum().item())
    totals["supervised"] += float(supervised.sum().item())
    changed = (fused_prediction != masks["prediction_a"]) & valid
    totals["changed"] += float(changed.sum().item())
    totals["changed_positive"] += float((changed & positive).sum().item())
    totals["changed_negative"] += float((changed & negative).sum().item())
    totals["changed_both_wrong"] += float(
        (changed & masks["both_wrong"]).sum().item()
    )

    gate_2d = gate.squeeze(1)
    weight_2d = weight_b.squeeze(1)
    totals["gate_valid_sum"] += float(gate_2d[valid].sum().item())
    totals["gate_disagreement_sum"] += float(
        gate_2d[disagreement].sum().item()
    )
    totals["gate_positive_sum"] += float(gate_2d[positive].sum().item())
    totals["gate_negative_sum"] += float(gate_2d[negative].sum().item())
    totals["weight_valid_sum"] += float(weight_2d[valid].sum().item())
    totals["weight_disagreement_sum"] += float(
        weight_2d[disagreement].sum().item()
    )
    totals["near_cap"] += float(((gate_2d >= 0.95) & valid).sum().item())


def _fusion_summary(totals: Dict[str, float]) -> Dict[str, float]:
    return {
        "disagreement_frac": _ratio(totals["disagreement"], totals["valid"]),
        "route_positive_frac": _ratio(totals["positive"], totals["supervised"]),
        "changed_frac": _ratio(totals["changed"], totals["valid"]),
        "changed_precision": _ratio(totals["changed_positive"], totals["changed"]),
        "rescue_recall": _ratio(totals["changed_positive"], totals["positive"]),
        "false_change_rate": _ratio(totals["changed_negative"], totals["negative"]),
        "changed_both_wrong_rate": _ratio(
            totals["changed_both_wrong"], totals["both_wrong_disagreement"]
        ),
        "gate_prob_positive": _ratio(
            totals["gate_positive_sum"], totals["positive"]
        ),
        "gate_prob_negative": _ratio(
            totals["gate_negative_sum"], totals["negative"]
        ),
        "gate_prob_valid_mean": _ratio(totals["gate_valid_sum"], totals["valid"]),
        "gate_prob_disagreement_mean": _ratio(
            totals["gate_disagreement_sum"], totals["disagreement"]
        ),
        "weight_b_valid_mean": _ratio(
            totals["weight_valid_sum"], totals["valid"]
        ),
        "weight_b_disagreement_mean": _ratio(
            totals["weight_disagreement_sum"], totals["disagreement"]
        ),
        "near_cap_frac": _ratio(totals["near_cap"], totals["valid"]),
    }


def _new_fusion_totals() -> Dict[str, float]:
    keys = (
        "valid",
        "disagreement",
        "positive",
        "negative",
        "both_wrong_disagreement",
        "supervised",
        "changed",
        "changed_positive",
        "changed_negative",
        "changed_both_wrong",
        "gate_valid_sum",
        "gate_disagreement_sum",
        "gate_positive_sum",
        "gate_negative_sum",
        "weight_valid_sum",
        "weight_disagreement_sum",
        "near_cap",
    )
    return {key: 0.0 for key in keys}


def _open_image(path: str, resize: int) -> Image.Image:
    return Image.open(path).convert("RGB").resize(
        (resize, resize), Image.BILINEAR
    )


@torch.inference_mode()
def frozen_logits(
    tcsb: loveda_b3.FrozenTCSB,
    tsdb: loveda_b3.TSDB,
    image: Image.Image,
    resize: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits_a = tcsb.logits(image, resize).float()
    logits_b = tsdb.forward_logits(image).float()
    if logits_a.ndim != 4 or logits_a.shape[1] != NUM_CLASSES:
        raise RuntimeError(
            f"TCSB must return [B,{NUM_CLASSES},H,W], got {tuple(logits_a.shape)}"
        )
    if logits_b.ndim != 4 or logits_b.shape[1] != NUM_CLASSES:
        raise RuntimeError(
            f"TSDB must return [B,{NUM_CLASSES},H,W], got {tuple(logits_b.shape)}"
        )
    if logits_a.shape[-2:] != (resize, resize):
        logits_a = F.interpolate(
            logits_a,
            size=(resize, resize),
            mode="bilinear",
            align_corners=False,
        )
    if logits_b.shape[-2:] != (resize, resize):
        logits_b = F.interpolate(
            logits_b,
            size=(resize, resize),
            mode="bilinear",
            align_corners=False,
        )
    return logits_a, logits_b


def train_one_epoch(
    router: SelectiveRescueRouter,
    tcsb: loveda_b3.FrozenTCSB,
    tsdb: loveda_b3.TSDB,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    resize: int,
    epoch: int,
    args,
) -> Dict[str, float]:
    router.train()
    tsdb.eval()
    tsdb.sam_model.eval()
    running = OrderedDict(
        loss_total=0.0,
        loss_ce=0.0,
        loss_xded=0.0,
        loss_cons=0.0,
        lambda_xded_eff=0.0,
        batches=0.0,
        augmented_batches=0.0,
    )
    totals = _new_fusion_totals()
    num_batches = max(len(loader), 1)

    for batch_index, batch in enumerate(
        tqdm(loader, desc="Fusion training", leave=False)
    ):
        image_original = _open_image(batch["img_path"][0], resize)
        target = batch["gt"][0].to(device, non_blocking=True)

        logits_a_original, logits_b_original = frozen_logits(
            tcsb,
            tsdb,
            image_original,
            resize,
        )
        masks_original = relation_masks(
            logits_a_original,
            logits_b_original,
            target,
        )
        features_original = build_router_features(
            logits_a_original,
            logits_b_original,
        ).detach()

        optimizer.zero_grad(set_to_none=True)
        route_logits_original = router(features_original)
        fused_original, weight_original, gate_original = bounded_soft_fusion(
            logits_a_original,
            logits_b_original,
            route_logits_original,
            max_b_weight=args.max_b_weight,
        )
        loss_ce_original = fused_cross_entropy(fused_original, target)
        loss_xded_original = selective_xded_loss(
            logits_a_original,
            logits_b_original,
            weight_original,
            masks_original,
            temperature=args.xded_temperature,
            negative_cost=float(args.negative_cost),
        )

        use_augmented = random.random() < float(args.router_aug_prob)
        if use_augmented:
            image_augmented, _, _ = domain_randomize(
                image_original,
                strength=float(args.router_aug_strength),
            )
            logits_a_augmented, logits_b_augmented = frozen_logits(
                tcsb,
                tsdb,
                image_augmented,
                resize,
            )
            masks_augmented = relation_masks(
                logits_a_augmented,
                logits_b_augmented,
                target,
            )
            features_augmented = build_router_features(
                logits_a_augmented,
                logits_b_augmented,
            ).detach()
            route_logits_augmented = router(features_augmented)
            fused_augmented, weight_augmented, gate_augmented = bounded_soft_fusion(
                logits_a_augmented,
                logits_b_augmented,
                route_logits_augmented,
                max_b_weight=args.max_b_weight,
            )
            loss_ce_augmented = fused_cross_entropy(fused_augmented, target)
            loss_xded_augmented = selective_xded_loss(
                logits_a_augmented,
                logits_b_augmented,
                weight_augmented,
                masks_augmented,
                temperature=args.xded_temperature,
                negative_cost=float(args.negative_cost),
            )
            loss_ce = 0.5 * (loss_ce_original + loss_ce_augmented)
            loss_xded = 0.5 * (loss_xded_original + loss_xded_augmented)
            loss_cons = router_consistency_loss(
                gate_original,
                gate_augmented,
                masks_original["disagreement"],
                masks_augmented["disagreement"],
            )
            running["augmented_batches"] += 1.0
        else:
            loss_ce = loss_ce_original
            loss_xded = loss_xded_original
            loss_cons = gate_original.sum() * 0.0

        if args.xded_warmup_epochs > 0:
            warmup_progress = (
                (epoch - 1) + float(batch_index + 1) / float(num_batches)
            ) / float(args.xded_warmup_epochs)
            lambda_xded_eff = float(args.lambda_xded) * min(
                max(warmup_progress, 0.0),
                1.0,
            )
        else:
            lambda_xded_eff = float(args.lambda_xded)

        loss_total = (
            loss_ce
            + lambda_xded_eff * loss_xded
            + float(args.lambda_cons) * loss_cons
        )
        loss_total.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(router.parameters(), args.grad_clip)
        optimizer.step()

        with torch.no_grad():
            fused_prediction = fused_original.argmax(dim=1)
            _accumulate_fusion_stats(
                totals,
                masks_original,
                fused_prediction,
                gate_original,
                weight_original,
            )
        running["loss_total"] += float(loss_total.detach().cpu())
        running["loss_ce"] += float(loss_ce.detach().cpu())
        running["loss_xded"] += float(loss_xded.detach().cpu())
        running["loss_cons"] += float(loss_cons.detach().cpu())
        running["lambda_xded_eff"] += float(lambda_xded_eff)
        running["batches"] += 1.0

        del (
            logits_a_original,
            logits_b_original,
            features_original,
            route_logits_original,
            fused_original,
            weight_original,
            gate_original,
            loss_total,
        )

    batches = max(running["batches"], 1.0)
    summary = {
        "loss_total": running["loss_total"] / batches,
        "loss_ce": running["loss_ce"] / batches,
        "loss_xded": running["loss_xded"] / batches,
        "loss_cons": running["loss_cons"] / batches,
        "lambda_xded_eff": running["lambda_xded_eff"] / batches,
        "trained_batches": running["batches"],
        "augmented_batches": running["augmented_batches"],
        "router_aug_prob": float(args.router_aug_prob),
        "max_b_weight": float(args.max_b_weight),
    }
    summary.update(_fusion_summary(totals))
    return summary


@torch.inference_mode()
def evaluate(
    router: SelectiveRescueRouter,
    tcsb: loveda_b3.FrozenTCSB,
    tsdb: loveda_b3.TSDB,
    loader: DataLoader,
    device: torch.device,
    resize: int,
    max_b_weight: float,
    description: str,
) -> Dict:
    router.eval()
    tsdb.eval()
    tsdb.sam_model.eval()
    confusions = {
        name: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        for name in ("a", "b", "fusion", "oracle")
    }
    totals = _new_fusion_totals()

    for batch in tqdm(loader, desc=description, leave=False):
        image = _open_image(batch["img_path"][0], resize)
        target = batch["gt"][0].to(device, non_blocking=True)
        logits_a, logits_b = frozen_logits(
            tcsb,
            tsdb,
            image,
            resize,
        )
        masks = relation_masks(logits_a, logits_b, target)
        features = build_router_features(logits_a, logits_b)
        route_logits = router(features)
        probability_fused, weight_b, gate = bounded_soft_fusion(
            logits_a,
            logits_b,
            route_logits,
            max_b_weight=max_b_weight,
        )
        fused = probability_fused.argmax(dim=1)
        _accumulate_fusion_stats(
            totals,
            masks,
            fused,
            gate,
            weight_b,
        )

        oracle = torch.where(
            masks["positive"],
            masks["prediction_b"],
            masks["prediction_a"],
        )
        valid = masks["valid"]
        target_valid = masks["target"][valid].cpu().numpy()
        predictions = {
            "a": masks["prediction_a"][valid].cpu().numpy(),
            "b": masks["prediction_b"][valid].cpu().numpy(),
            "fusion": fused[valid].cpu().numpy(),
            "oracle": oracle[valid].cpu().numpy(),
        }
        for name, prediction in predictions.items():
            _update_confusion(confusions[name], target_valid, prediction)

    output = {"fusion_stats": _fusion_summary(totals)}
    for name, confusion in confusions.items():
        miou, ious, pixel_accuracy = _scores(confusion)
        output[name] = {
            "mIoU": miou,
            "pixel_accuracy": pixel_accuracy,
            "per_class_iou": dict(zip(CLASS_NAMES, ious)),
            "confusion": confusion.tolist(),
        }
    output["fusion_gain_over_a"] = (
        output["fusion"]["mIoU"] - output["a"]["mIoU"]
    )
    output["oracle_gain_over_a"] = (
        output["oracle"]["mIoU"] - output["a"]["mIoU"]
    )
    return output


def print_evaluation(label: str, result: Dict) -> None:
    log_fusion_evaluation(label, result)


def save_router_checkpoint(path, router, optimizer, epoch, args,
                           tsdb_checkpoint, target_monitor):
    torch.save({
        "epoch": int(epoch),
        "router_state_dict": {k: v.detach().cpu() for k, v in router.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "class_names": CLASS_NAMES,
        "num_classes": NUM_CLASSES,
        "tcsb_lora_checkpoint": args.tcsb_lora_ckpt,
        "tsdb_checkpoint": args.tsdb_ckpt,
        "max_b_weight": float(args.max_b_weight),
    }, path)


def resolve_models(args, device: torch.device):
    checkpoint_b = _safe_torch_load(args.tsdb_ckpt)
    saved_b = checkpoint_b["args"]
    if args.resize is None:
        args.resize = int(saved_b.get("resize", 512))
    if args.global_scale is None:
        args.global_scale = 0.8
    class_weights = [float(v) for v in str(saved_b.get(
        "class_weights", "0.8,1.1,1.1,1.1,1.5,1.5,1.5")).split(",")]
    tcsb = loveda_b3.FrozenTCSB(args.sam_ckpt, args.tcsb_lora_ckpt, device)
    tsdb = loveda_b3.TSDB(
        sam_ckpt=args.sam_ckpt,
        device=device,
        resize=args.resize,
        feat_channels=int(saved_b.get("embed_dim", 256) or 256),
        out_channels=int(saved_b.get("embed_dim", 256) or 256),
        num_queries=int(saved_b.get("num_queries", 100) or 100),
        decoder_layers=int(saved_b.get("decoder_layers", 6) or 6),
        num_heads=int(saved_b.get("num_heads", 8) or 8),
        use_gse=True,
        query_mode=str(saved_b.get("query_mode", "add") or "add"),
        gse_base_channels=int(saved_b.get("gse_base_channels", 64) or 64),
        replace_query_feat=bool(saved_b.get("replace_query_feat", False)),
        no_object_weight=float(saved_b.get("no_object_weight", 0.1) or 0.1),
        class_weights=class_weights,
        use_pixel_aux=bool(saved_b.get("use_pixel_aux", False)),
        lambda_pixel=float(saved_b.get("lambda_pixel", 0.4) or 0.4),
        pixel_fusion_weight=float(
            saved_b.get("pixel_fusion_weight", 0.5) or 0.5
        ),
        use_earth_adapter=True,
        earth_dim=int(saved_b.get("earth_dim", 24) or 24),
        earth_adapter_start=int(saved_b.get("earth_adapter_start", 0) or 0),
        earth_fft_start=int(saved_b.get("earth_fft_start", 24) or 24),
        earth_cutoff_ratio=float(
            saved_b.get("earth_cutoff_ratio", 0.3) or 0.3
        ),
        earth_scale=float(saved_b.get("earth_scale", 0.1) or 0.1),
        use_structure_injection=True,
        structure_scale=float(saved_b.get("structure_scale", 0.05) or 0.05),
    ).to(device)
    missing, unexpected = tsdb.load_state_dict(
        checkpoint_b["model_state_dict"], strict=False
    )
    bad_missing = [key for key in missing if not key.startswith("sam_model.")]
    if bad_missing or unexpected:
        raise RuntimeError(
            "TSDB state mismatch: "
            f"bad_missing={bad_missing[:20]}, unexpected={unexpected[:20]}"
        )
    for parameter in tsdb.parameters():
        parameter.requires_grad = False
    tsdb.eval()
    tsdb.sam_model.eval()
    return tcsb, tsdb, checkpoint_b


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate bounded soft fusion of the two frozen experts.")
    parser.add_argument("--tcsb-lora-ckpt", required=True)
    parser.add_argument("--tsdb-ckpt", required=True)
    parser.add_argument("--sam-ckpt", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resize", type=int, default=None)
    parser.add_argument(
        "--global-scale",
        type=float,
        default=None,
        help=(
            "TCSB LoRA scale. It must match the checkpoint's fixed "
            "eval_global_lora_scale=0.8."
        ),
    )

    parser.add_argument(
        "--train-img-dir",
        default="data/LoveDA/Train/Urban/images_png",
    )
    parser.add_argument(
        "--train-ann-dir",
        default="data/LoveDA/Train/Urban/masks_png",
    )
    parser.add_argument(
        "--target-img-dir",
        default="data/LoveDA/Val/Rural/images_png",
    )
    parser.add_argument(
        "--target-ann-dir",
        default="data/LoveDA/Val/Rural/masks_png",
    )
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-target", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--router-scale", type=float, default=0.5)
    parser.add_argument(
        "--negative-cost",
        type=float,
        default=1.5,
        help="Extra XDED weight where A is correct and B is wrong.",
    )
    parser.add_argument(
        "--max-b-weight",
        type=float,
        default=0.20,
        help="Upper bound of continuous TSDB probability weight.",
    )
    parser.add_argument("--lambda-xded", type=float, default=0.5)
    parser.add_argument("--xded-temperature", type=float, default=4.0)
    parser.add_argument("--xded-warmup-epochs", type=float, default=1.0)
    parser.add_argument("--lambda-cons", type=float, default=0.1)
    parser.add_argument("--router-aug-prob", type=float, default=0.5)
    parser.add_argument("--router-aug-strength", type=float, default=0.5)

    parser.add_argument(
        "--eval-interval",
        type=int,
        default=3,
        help="Keep at 3 for the locked three-epoch U->R confirmation run.",
    )
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume-router-ckpt", default="")
    parser.add_argument("--target-mask-encoding", choices=["loveda_raw", "train_ids"], default="loveda_raw")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--work-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.eval_interval < 1 or args.save_interval < 1:
        raise ValueError("eval/save intervals must be >= 1")
    if args.negative_cost < 1.0:
        raise ValueError("--negative-cost must be >= 1.0")
    if not 0.0 < args.max_b_weight < 1.0:
        raise ValueError("--max-b-weight must be in (0, 1)")
    if args.lambda_xded < 0.0 or args.lambda_cons < 0.0:
        raise ValueError("loss weights must be >= 0")
    if args.xded_temperature <= 0.0:
        raise ValueError("--xded-temperature must be > 0")
    if args.xded_warmup_epochs < 0.0:
        raise ValueError("--xded-warmup-epochs must be >= 0")
    if not 0.0 <= args.router_aug_prob <= 1.0:
        raise ValueError("--router-aug-prob must be in [0, 1]")
    if not 0.0 <= args.router_aug_strength <= 1.0:
        raise ValueError("--router-aug-strength must be in [0, 1]")

    work_dir = Path(args.work_dir)
    os.makedirs(work_dir, exist_ok=True)
    loveda_base.set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tcsb, tsdb, checkpoint_b = resolve_models(args, device)
    router = SelectiveRescueRouter(
        num_classes=NUM_CLASSES,
        hidden_dim=args.hidden_dim,
        router_scale=args.router_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(
        router.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    start_epoch = 1
    if args.resume_router_ckpt:
        resume = _safe_torch_load(args.resume_router_ckpt)
        router.load_state_dict(resume["router_state_dict"])
        if "optimizer_state_dict" in resume and not args.eval_only:
            optimizer.load_state_dict(resume["optimizer_state_dict"])
        start_epoch = int(resume.get("epoch", 0)) + 1

    train_set = None if args.eval_only else loveda_base.LoveDASingleImageDataset(
        args.train_img_dir,
        args.train_ann_dir,
        args.resize,
        args.max_train,
    )
    target_set = loveda_base.LoveDASingleImageDataset(
        args.target_img_dir,
        args.target_ann_dir,
        args.resize,
        args.max_target,
        mask_encoding=args.target_mask_encoding,
    )
    if (train_set is not None and len(train_set) == 0) or len(target_set) == 0:
        raise RuntimeError("Training or target-monitor dataset contains no pairs")
    train_loader = None if args.eval_only else DataLoader(
        train_set,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    target_loader = DataLoader(
        target_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    if args.eval_only:
        print(f"[Fusion] Evaluation | images={len(target_set)}", flush=True)
    else:
        print(f"[Fusion] Training | epochs={args.epochs} "
              f"train_images={len(train_set)} val_images={len(target_set)}", flush=True)
    print(f"[Fusion] Output: {work_dir}", flush=True)

    if args.eval_only:
        if not args.resume_router_ckpt:
            raise ValueError("--eval-only requires --resume-router-ckpt")
        result = evaluate(
            router,
            tcsb,
            tsdb,
            target_loader,
            device,
            args.resize,
            args.max_b_weight,
            "Evaluation",
        )
        print_evaluation("Fusion", result)
        with open(work_dir / "eval_only.json", "w", encoding="utf-8") as file:
            json.dump(result, file, indent=2, ensure_ascii=False)
        return

    history = []
    last_target_result = None
    for epoch in range(start_epoch, args.epochs + 1):
        train_log = train_one_epoch(
            router,
            tcsb,
            tsdb,
            train_loader,
            optimizer,
            device,
            args.resize,
            epoch,
            args,
        )
        log_epoch("Fusion", epoch, args.epochs, train_log)


        save_this_epoch = (
            epoch % args.save_interval == 0 or epoch == args.epochs
        )
        recovery_path = work_dir / f"soft_fusion_epoch_{epoch}.pth"
        if save_this_epoch:
            save_router_checkpoint(
                recovery_path,
                router,
                optimizer,
                epoch,
                args,
                checkpoint_b,
                None,
            )

        target_result = None
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            target_result = evaluate(
                router,
                tcsb,
                tsdb,
                target_loader,
                device,
                args.resize,
                args.max_b_weight,
                "Validation",
            )
            last_target_result = target_result

        record = {
            "epoch": epoch,
            "train_log": train_log,
            "target_monitor": target_result,
        }
        history.append(record)
        with open(work_dir / "history.json", "w", encoding="utf-8") as file:
            json.dump(history, file, indent=2, ensure_ascii=False)

        if save_this_epoch:
            save_router_checkpoint(
                recovery_path,
                router,
                optimizer,
                epoch,
                args,
                checkpoint_b,
                target_result,
            )

    final_path = work_dir / "final_router.pth"
    save_router_checkpoint(
        final_path,
        router,
        optimizer,
        args.epochs,
        args,
        checkpoint_b,
        last_target_result,
    )
    print(f"[Fusion] Saved: {final_path}", flush=True)
    if last_target_result is not None:
        print_evaluation("Fusion", last_target_result)


if __name__ == "__main__":
    main()
