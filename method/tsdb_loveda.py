from __future__ import annotations

from method.logging_utils import log_epoch

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader
from method.logging_utils import progress as tqdm

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from method.tsdb_backbone import loveda as base


VARIANT_CONFIG = {"b3": dict(appearance=True, rescue=True, structure=True)}


EXPERT_A_PROMPTS = [
    "building",
    "road",
    "water",
    "barren land",
    "forest",
    "agricultural land",
]


def _rgb01(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _rgb_pil(x: np.ndarray) -> Image.Image:
    return Image.fromarray(
        np.round(np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB"
    )


def _smooth_field(height: int, width: int, grid: int = 6) -> np.ndarray:
    small = np.random.uniform(-1.0, 1.0, size=(grid, grid)).astype(np.float32)
    field = Image.fromarray(np.round((small + 1.0) * 127.5).astype(np.uint8), mode="L")
    field = field.resize((width, height), Image.BICUBIC)
    return np.asarray(field, dtype=np.float32) / 127.5 - 1.0


def _spectral_shift(x: np.ndarray, strength: float) -> np.ndarray:

    s = float(strength)
    if random.random() < 0.45:
        r, g, b = x[..., 0], x[..., 1], x[..., 2]
        green_excess = np.maximum(2.0 * g - r - b, 0.0)
        pseudo_nir = np.clip(0.30 * r + 0.55 * g + 0.15 * b + 0.30 * green_excess, 0.0, 1.0)
        pseudo_irrg = np.stack([pseudo_nir, r, g], axis=-1)
        amount = random.uniform(0.45, 0.80) * s
        x = (1.0 - amount) * x + amount * pseudo_irrg
    else:
        rnd = np.random.gamma(1.5, 1.0, size=(3, 3)).astype(np.float32)
        rnd /= np.maximum(rnd.sum(axis=1, keepdims=True), 1e-6)
        amount = random.uniform(0.30, 0.65) * s
        response = (1.0 - amount) * np.eye(3, dtype=np.float32) + amount * rnd
        x = np.einsum("hwc,dc->hwd", x, response)

    gain = np.random.uniform(1.0 - 0.30 * s, 1.0 + 0.30 * s, (1, 1, 3)).astype(np.float32)
    bias = np.random.uniform(-0.08 * s, 0.08 * s, (1, 1, 3)).astype(np.float32)
    gamma = np.random.uniform(max(0.55, 1.0 - 0.38 * s), 1.0 + 0.45 * s, (1, 1, 3)).astype(np.float32)
    return np.power(np.clip(x * gain + bias, 0.0, 1.0), gamma)


def _weather_shift(x: np.ndarray, strength: float) -> np.ndarray:

    s = float(strength)
    h, w, _ = x.shape
    field = _smooth_field(h, w, grid=random.choice((4, 5, 6)))

    if random.random() < 0.60:


        base_haze = random.uniform(0.06, 0.28) * s
        transmission = np.clip(1.0 - base_haze * (1.0 + 0.45 * field), 0.55, 1.0)
        airlight = np.random.uniform(0.82, 1.0, size=(1, 1, 3)).astype(np.float32)
        x = x * transmission[..., None] + airlight * (1.0 - transmission[..., None])

    illumination = 1.0 + random.uniform(0.08, 0.28) * s * field[..., None]
    temperature = np.array(
        [random.uniform(1.0 - 0.15 * s, 1.0 + 0.18 * s),
         random.uniform(1.0 - 0.06 * s, 1.0 + 0.06 * s),
         random.uniform(1.0 - 0.18 * s, 1.0 + 0.15 * s)],
        dtype=np.float32,
    ).reshape(1, 1, 3)
    return x * illumination * temperature


def _sensor_shift(x: np.ndarray, strength: float) -> np.ndarray:

    s = float(strength)
    h, w, _ = x.shape
    img = _rgb_pil(x)


    factor = 1.0 + random.uniform(0.30, 1.40) * s
    small = (max(8, round(w / factor)), max(8, round(h / factor)))
    img = img.resize(small, Image.BILINEAR).resize((w, h), Image.BILINEAR)
    if random.random() < 0.65:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.15, 0.85) * s))
    x = _rgb01(img)
    sigma = random.uniform(0.003, 0.018) * s


    noise_scale = sigma * np.sqrt(np.clip(x, 0.05, 1.0)) + 0.25 * sigma
    return x + np.random.normal(0.0, 1.0, size=x.shape).astype(np.float32) * noise_scale


def _frequency_shift(x: np.ndarray, strength: float) -> np.ndarray:

    s = float(strength)
    h, w, _ = x.shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    low_cut = min(h, w) * random.uniform(0.035, 0.10)
    high_cut = min(h, w) * random.uniform(0.22, 0.38)
    low_mask = (radius <= low_cut)
    high_mask = (radius >= high_cut)
    low_gain = random.uniform(1.0 - 0.55 * s, 1.0 + 0.55 * s)
    high_gain = random.uniform(1.0 - 0.65 * s, 1.0 + 0.30 * s)
    out = np.empty_like(x)
    for channel in range(3):
        spectrum = np.fft.fftshift(np.fft.fft2(x[..., channel]))
        amplitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        dc = amplitude[cy, cx].copy()
        amplitude[low_mask] *= low_gain * random.uniform(0.94, 1.06)
        amplitude[high_mask] *= high_gain
        amplitude[cy, cx] = dc
        restored = amplitude * np.exp(1j * phase)
        out[..., channel] = np.real(np.fft.ifft2(np.fft.ifftshift(restored)))
    return out


def _edge_correlation(a: np.ndarray, b: np.ndarray) -> float:
    gray_a = a.mean(axis=-1)
    gray_b = b.mean(axis=-1)
    ay, ax = np.gradient(gray_a)
    by, bx = np.gradient(gray_b)
    edge_a = np.sqrt(ax * ax + ay * ay).reshape(-1)
    edge_b = np.sqrt(bx * bx + by * by).reshape(-1)
    if edge_a.std() < 1e-7 or edge_b.std() < 1e-7:
        return 1.0
    return float(np.corrcoef(edge_a, edge_b)[0, 1])


def domain_randomize(
    image: Image.Image,
    strength: float = 0.75,
) -> Tuple[Image.Image, Dict[str, float], str]:

    s = float(np.clip(strength, 0.0, 1.0))
    clean = _rgb01(image)
    x = clean.copy()
    operations = {
        "spectral": _spectral_shift,
        "weather": _weather_shift,
        "sensor": _sensor_shift,
        "frequency": _frequency_shift,
    }
    names = list(operations)
    primary = random.choices(names, weights=(0.38, 0.24, 0.20, 0.18), k=1)[0]
    x = operations[primary](x, s)
    used = [primary]
    if random.random() < 0.40:
        secondary = random.choice([name for name in names if name != primary])
        x = operations[secondary](np.clip(x, 0.0, 1.0), 0.40 * s)
        used.append(secondary)

    x = np.clip(x, 0.0, 1.0)
    delta = x - clean
    stats = {
        "aug_mad": float(np.abs(delta).mean()),
        "aug_rms": float(np.sqrt(np.square(delta).mean())),
        "aug_channel_mean_shift": float(
            np.abs(x.mean(axis=(0, 1)) - clean.mean(axis=(0, 1))).mean()
        ),
        "aug_saturated_frac": float(((x <= 1e-4) | (x >= 1.0 - 1e-4)).mean()),
        "aug_edge_corr": _edge_correlation(clean, x),
    }
    return _rgb_pil(x), stats, "+".join(used)


def _block_id(module_name: str) -> Optional[int]:
    match = re.search(r"(?:blocks|layers)\.(\d+)", module_name)
    return int(match.group(1)) if match else None


def _parent_and_child(root: nn.Module, module_name: str):
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _lora_targets(model: nn.Module, blocks: Sequence[int]) -> List[str]:
    keep = {int(x) for x in blocks}
    targets = []
    for name, module in model.named_modules():
        lname = name.lower()
        if not isinstance(module, nn.Linear):
            continue
        if "backbone.vision_backbone.trunk.blocks." not in lname or "attn" not in lname:
            continue
        if _block_id(lname) in keep and any(
            token in lname for token in ("q", "k", "v", "proj", "out", "fc")
        ):
            targets.append(name)
    return sorted(set(targets))


class FrozenTCSB:


    def __init__(self, sam_ckpt: str, lora_ckpt: str, device: torch.device):
        raw = torch.load(lora_ckpt, map_location="cpu", weights_only=False)
        state = raw["lora_state_dict"]
        rank_map = {int(k): int(v) for k, v in raw["final_rank_map"].items()}
        blocks = raw["candidate_blocks"]
        saved = raw["args"]
        max_r = max(value.shape[0] for key, value in state.items() if key.endswith("lora_A"))
        model = build_sam3_image_model(checkpoint_path=sam_ckpt, device=str(device), eval_mode=True)
        for name in _lora_targets(model, blocks):
            parent, child = _parent_and_child(model, name)
            wrapped = base.DynamicLoRALinear(
                getattr(parent, child), max_r=max_r,
                alpha_mult=float(saved.get("alpha_multiplier", 2.0)), dropout=0.0,
                block_id=_block_id(name), init_active_r=max_r,
            ).to(device)
            setattr(parent, child, wrapped)
        model.load_state_dict(state, strict=False)
        scale = float(raw.get("eval_global_lora_scale", raw["global_lora_scale"]))
        for module in model.modules():
            if isinstance(module, base.DynamicLoRALinear):
                with torch.no_grad():
                    module.active_r.fill_(rank_map[module.block_id])
                    module.global_scale.fill_(scale)
        for parameter in model.parameters():
            parameter.requires_grad = False
        self.model = model.to(device).eval()
        self.processor = Sam3Processor(model=self.model, device=str(device))
        self.device = device

    @torch.no_grad()
    def logits(self, image: Image.Image, resize: int) -> torch.Tensor:
        height, width = image.size[1], image.size[0]
        state = self.processor.set_image(image)
        foreground = []
        for prompt in EXPERT_A_PROMPTS:
            self.processor.reset_all_prompts(state)
            state = self.processor.set_text_prompt(state=state, prompt=prompt)
            sem = state.get("semantic_mask_logits")
            if sem is None:
                sem = torch.full(
                    (1, 1, height, width), -1e4, device=self.device, dtype=torch.float32
                )
            elif sem.ndim == 3:
                sem = sem.unsqueeze(1)
            sem = sem.float()
            if sem.shape[-2:] != (resize, resize):
                sem = F.interpolate(
                    sem, size=(resize, resize), mode="bilinear", align_corners=False
                )
            foreground.append(sem)
        foreground_logits = torch.cat(foreground, dim=1)
        background = -foreground_logits.amax(dim=1, keepdim=True)


        return torch.cat([background, foreground_logits], dim=1)


class DomainStableStructureInjector(nn.Module):


    def __init__(self, channels=(256, 256, 256, 256), init_scale: float = 0.05):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-1, -2).contiguous()
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(3, c, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(32, c),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(c, c, kernel_size=1, bias=False),
                )
                for c in channels
            ]
        )

        gate = torch.full((len(channels),), float(init_scale)).atanh()
        self.gates = nn.Parameter(gate)

    def _structure_map(self, image_tensor: torch.Tensor) -> torch.Tensor:
        gray = image_tensor.mean(dim=1, keepdim=True)
        mean = gray.mean(dim=(-2, -1), keepdim=True)
        std = gray.std(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
        gray = (gray - mean) / std
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        mag = torch.sqrt(gx.square() + gy.square() + 1e-6)
        mag_scale = mag.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
        mag = mag / mag_scale
        return torch.cat([gray, gx / mag_scale, mag], dim=1)

    def forward(self, feats, image_tensor: torch.Tensor):
        structure = self._structure_map(image_tensor)
        outputs = []
        for level, (feat, encoder) in enumerate(zip(feats, self.encoders)):
            signal = F.interpolate(
                structure, size=feat.shape[-2:], mode="bilinear", align_corners=False
            )
            residual = encoder(signal)
            outputs.append(feat + torch.tanh(self.gates[level]) * residual)
        return outputs

    def info(self):
        return {
            "enabled": True,
            "gate_values": torch.tanh(self.gates).detach().cpu().tolist(),
            "signal": "per-image normalized gray + Sobel-x + edge magnitude",
        }


class TSDB(base.FrozenSAMReinMask2Former):
    def __init__(self, *args, use_structure_injection=False, structure_scale=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_structure_injection = bool(use_structure_injection)
        self.structure_injector = None
        if self.use_structure_injection:
            self.structure_injector = DomainStableStructureInjector(
                channels=tuple(self.proj_channels), init_scale=structure_scale
            )

    def _build_features_and_query(self, image_pil: Image.Image):
        feats, query_embed = super()._build_features_and_query(image_pil)
        if self.structure_injector is not None:
            image_tensor = base.pil_to_tensor01(image_pil, self.device0)
            feats = self.structure_injector(feats, image_tensor)
        return feats, query_embed

    @staticmethod
    def _semantic_logits(
        cls_pred: torch.Tensor,
        mask_pred: torch.Tensor,
        size: Tuple[int, int],
    ) -> torch.Tensor:

        cls_score = F.softmax(cls_pred, dim=-1)[..., : base.NUM_CLASSES]
        mask_prob = mask_pred.sigmoid()
        seg = torch.einsum("bqc,bqhw->bchw", cls_score, mask_prob)
        logits = torch.log(seg.clamp_min(1e-6))
        return F.interpolate(
            logits, size=size, mode="bilinear", align_corners=False
        )

    def loss_and_logits(
        self, image_pil: Image.Image, gt: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:

        feats, query_embed = self._build_features_and_query(image_pil)
        sample = base.make_seg_data_sample(gt, self.resize)
        cls_pred_list, mask_pred_list = self.decode_head(
            (feats, query_embed), [sample]
        )


        gt_instances, img_metas = self.decode_head._seg_data_to_instance_data(
            [sample]
        )
        losses = self.decode_head.loss_by_feat(
            cls_pred_list, mask_pred_list, gt_instances, img_metas
        )
        logits = self._semantic_logits(
            cls_pred_list[-1],
            mask_pred_list[-1],
            (self.resize, self.resize),
        )

        if self.use_pixel_aux:
            pixel_logits = self.pixel_head(feats)
            if pixel_logits.shape[-2:] != logits.shape[-2:]:
                pixel_logits = F.interpolate(
                    pixel_logits,
                    size=logits.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            target = gt.long()
            if target.ndim == 2:
                target = target.unsqueeze(0)
            loss_pixel_raw = F.cross_entropy(
                pixel_logits, target, ignore_index=255
            )
            losses["loss_pixel_aux"] = self.lambda_pixel * loss_pixel_raw
            losses["pixel_aux_raw"] = loss_pixel_raw.detach()
            logits = logits + self.pixel_fusion_weight * pixel_logits

        return losses, logits

    def structure_info(self):
        if self.structure_injector is None:
            return {"enabled": False}
        return self.structure_injector.info()


def rescue_loss_and_metrics(
    logits_b: torch.Tensor,
    logits_a: torch.Tensor,
    gt: torch.Tensor,
    gamma: float,
    max_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:

    if gt.ndim == 2:
        target = gt.unsqueeze(0).long()
    elif gt.ndim == 3:
        target = gt.long()
    else:
        raise RuntimeError(f"Unexpected target shape: {tuple(gt.shape)}")
    if logits_b.shape[-2:] != target.shape[-2:]:
        logits_b = F.interpolate(logits_b, target.shape[-2:], mode="bilinear", align_corners=False)
    if logits_a.shape[-2:] != target.shape[-2:]:
        logits_a = F.interpolate(logits_a, target.shape[-2:], mode="bilinear", align_corners=False)

    valid = target != 255
    safe_target = target.masked_fill(~valid, 0)
    with torch.no_grad():
        prob_a = F.softmax(logits_a, dim=1)
        p_a_gt = prob_a.gather(1, safe_target.unsqueeze(1)).squeeze(1)
        weight = (1.0 - p_a_gt).pow(float(gamma)).clamp(max=float(max_weight))
        weight = weight * valid.float()

    ce_b = F.cross_entropy(logits_b, target, ignore_index=255, reduction="none")
    loss_rescue = (ce_b * weight).sum() / weight.sum().clamp_min(1e-6)

    with torch.no_grad():
        pred_a = logits_a.argmax(dim=1)
        pred_b = logits_b.argmax(dim=1)
        a_correct = (pred_a == target) & valid
        b_correct = (pred_b == target) & valid
        a_wrong = (~a_correct) & valid
        valid_n = valid.sum().clamp_min(1)
        a_wrong_n = a_wrong.sum().clamp_min(1)
        a_correct_n = a_correct.sum().clamp_min(1)
        metrics = {
            "rescue_weight_mean": float(weight[valid].mean().cpu()),
            "a_wrong_frac": float((a_wrong.sum() / valid_n).cpu()),
            "b_correct_on_a_wrong": float(((b_correct & a_wrong).sum() / a_wrong_n).cpu()),
            "a_correct_b_wrong": float(((a_correct & ~b_correct).sum() / a_correct_n).cpu()),
            "source_oracle_acc": float(((a_correct | b_correct).sum() / valid_n).cpu()),
        }
    return loss_rescue, metrics


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    resize,
    args,
    epoch: int,
    tcsb: Optional[FrozenTCSB] = None,
):
    model.train()
    model.sam_model.eval()
    running: Dict[str, float] = OrderedDict()
    count = 0

    for batch in tqdm(loader, desc="TSDB training", leave=False):
        image = Image.open(batch["img_path"][0]).convert("RGB")
        image = image.resize((resize, resize), Image.BILINEAR)
        gt = batch["gt"][0].to(device)
        optimizer.zero_grad(set_to_none=True)

        cfg = VARIANT_CONFIG[args.variant]
        appearance_active = cfg["appearance"] and epoch >= args.appearance_start_epoch
        image_aug = None
        logits_a = None
        if appearance_active:
            image_aug, aug_stats, aug_name = domain_randomize(
                image, strength=args.appearance_strength
            )

        if cfg["rescue"] and appearance_active:
            if tcsb is None:
                raise RuntimeError("B2/B3 require a restored frozen TCSB")
            if image_aug is None:
                raise RuntimeError("Rescue guidance requires the augmented view")


            logits_a = tcsb.logits(image_aug, resize).detach()

        clean_losses = model.loss(image, gt)
        clean_loss, logs = base.sum_loss_dict(clean_losses)


        aug_weight = float(args.appearance_seg_weight) if appearance_active else 0.0
        clean_coefficient = 1.0 / (1.0 + aug_weight)
        clean_objective = clean_coefficient * clean_loss
        clean_objective.backward()
        loss_total_value = float(clean_objective.detach().cpu())
        del clean_losses, clean_loss, clean_objective

        logits_b = None
        augmented_objective = None
        if appearance_active:
            if cfg["rescue"]:


                aug_losses, logits_b = model.loss_and_logits(image_aug, gt)
            else:
                aug_losses = model.loss(image_aug, gt)
            aug_loss, _ = base.sum_loss_dict(aug_losses)


            augmented_objective = (
                aug_weight / (1.0 + aug_weight)
            ) * aug_loss
            logs["loss_appearance_seg"] = float(aug_loss.detach().cpu())
            for key, value in aug_stats.items():
                logs[key] = float(value)
            for family in ("spectral", "weather", "sensor", "frequency"):
                logs[f"aug_used_{family}"] = float(family in aug_name.split("+"))

        if cfg["rescue"] and appearance_active:
            if logits_a is None or logits_b is None:
                raise RuntimeError("Rescue logits were not produced")


            rescue_raw, rescue_metrics = rescue_loss_and_metrics(
                logits_b,
                logits_a,
                gt,
                gamma=args.rescue_gamma,
                max_weight=args.max_rescue_weight,
            )
            ramp = min(1.0, epoch / max(1, int(args.rescue_ramp_epochs)))
            lambda_eff = float(args.lambda_rescue) * ramp
            loss_rescue = lambda_eff * rescue_raw
            augmented_objective = augmented_objective + loss_rescue
            logs["loss_rescue_raw"] = float(rescue_raw.detach().cpu())
            logs["loss_rescue"] = float(loss_rescue.detach().cpu())
            logs["lambda_rescue_eff"] = float(lambda_eff)
            logs.update(rescue_metrics)

        if cfg["appearance"] and not appearance_active:
            logs["appearance_off_until"] = float(args.appearance_start_epoch)
        if cfg["rescue"] and not appearance_active:
            logs["loss_rescue"] = 0.0

        if augmented_objective is not None:
            augmented_objective.backward()
            loss_total_value += float(augmented_objective.detach().cpu())
        if args.grad_clip > 0:
            trainable = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()

        logs["loss_total"] = loss_total_value
        for key, value in logs.items():
            running[key] = running.get(key, 0.0) + float(value)
        count += 1

    return {key: value / max(count, 1) for key, value in running.items()}


def save_checkpoint(
    model,
    optimizer,
    epoch,
    miou,
    ious,
    args,
    path: Path,
    role: str,
    tcsb: Optional[FrozenTCSB] = None,
):
    checkpoint = {
        "epoch": int(epoch),
        "checkpoint_role": role,
        "selection_policy": "fixed training epoch; target validation is monitoring only",
        "variant": args.variant,
        "variant_config": VARIANT_CONFIG[args.variant],
        "target_monitor_mIoU": None if miou is None else float(miou),
        "per_class_iou": (
            None
            if ious is None
            else {k: float(v) for k, v in zip(base.CLASS_NAMES, ious)}
        ),
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
            if not key.startswith("sam_model.")
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "decoder_info": model.info(),
        "structure_info": model.structure_info(),
        "class_names": base.CLASS_NAMES,
        "args": vars(args),
    }
    torch.save(checkpoint, path)


def parse_args():
    parser = argparse.ArgumentParser(description="TSDB with appearance augmentation, rescue supervision and structure injection.")
    parser.add_argument("--sam-ckpt", default="checkpoints/pretrained/sam3.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resize", type=int, default=512)
    parser.add_argument(
        "--train-img-dir",
        default="data/LoveDA/Train/Urban/images_png",
    )
    parser.add_argument(
        "--train-ann-dir",
        default="data/LoveDA/Train/Urban/masks_png",
    )
    parser.add_argument(
        "--val-img-dir",
        default="data/LoveDA/Val/Rural/images_png",
    )
    parser.add_argument(
        "--val-ann-dir",
        default="data/LoveDA/Val/Rural/masks_png",
    )
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--decoder-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--gse-base-channels", type=int, default=64)
    parser.add_argument("--query-mode", choices=["add", "replace"], default="add")
    parser.add_argument("--replace-query-feat", action="store_true")
    parser.add_argument("--no-object-weight", type=float, default=0.1)
    parser.add_argument(
        "--class-weights",
        default="0.8,1.1,1.1,1.1,1.5,1.5,1.5",
    )
    parser.add_argument("--use-pixel-aux", action="store_true")
    parser.add_argument("--lambda-pixel", type=float, default=0.4)
    parser.add_argument("--pixel-fusion-weight", type=float, default=0.5)

    parser.add_argument("--earth-dim", type=int, default=24)
    parser.add_argument("--earth-adapter-start", type=int, default=0)
    parser.add_argument("--earth-fft-start", type=int, default=24)
    parser.add_argument("--earth-cutoff-ratio", type=float, default=0.3)
    parser.add_argument("--earth-scale", type=float, default=0.1)

    parser.add_argument("--appearance-seg-weight", type=float, default=0.70)
    parser.add_argument("--appearance-start-epoch", type=int, default=1)
    parser.add_argument("--appearance-strength", type=float, default=0.75)
    parser.add_argument("--tcsb-lora-ckpt", default="")
    parser.add_argument("--lambda-rescue", type=float, default=0.50)
    parser.add_argument("--rescue-gamma", type=float, default=2.0)
    parser.add_argument("--max-rescue-weight", type=float, default=0.80)
    parser.add_argument("--rescue-ramp-epochs", type=int, default=2)
    parser.add_argument("--structure-scale", type=float, default=0.05)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--resume-ckpt", default="")
    parser.add_argument("--work-dir", default="")
    parser.set_defaults(variant="b3")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.eval_interval < 1 or args.save_interval < 1:
        raise ValueError("eval/save intervals must be >= 1")
    if not 0.0 <= args.appearance_strength <= 1.0:
        raise ValueError("--appearance-strength must be in [0, 1]")
    if not 0.0 < args.max_rescue_weight <= 1.0:
        raise ValueError("--max-rescue-weight must be in (0, 1]")

    if not args.work_dir:
        args.work_dir = f"work_dirs/tsdb_{args.variant}_loveda_u2r"
    work_dir = Path(args.work_dir)
    os.makedirs(work_dir, exist_ok=True)

    base.set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    class_weights = [float(value) for value in args.class_weights.split(",")]
    if len(class_weights) != base.NUM_CLASSES:
        raise ValueError(f"Expected {base.NUM_CLASSES} class weights")

    train_set = base.LoveDASingleImageDataset(
        args.train_img_dir, args.train_ann_dir, args.resize, args.max_train
    )
    val_set = base.LoveDASingleImageDataset(
        args.val_img_dir, args.val_ann_dir, args.resize, args.max_val
    )
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)


    cfg = VARIANT_CONFIG[args.variant]
    if cfg["rescue"] and not args.tcsb_lora_ckpt:
        raise ValueError("B2/B3 require --tcsb-lora-ckpt")
    model = TSDB(
        sam_ckpt=args.sam_ckpt,
        device=device,
        resize=args.resize,
        feat_channels=args.embed_dim,
        out_channels=args.embed_dim,
        num_queries=args.num_queries,
        decoder_layers=args.decoder_layers,
        num_heads=args.num_heads,
        use_gse=True,
        query_mode=args.query_mode,
        gse_base_channels=args.gse_base_channels,
        replace_query_feat=args.replace_query_feat,
        no_object_weight=args.no_object_weight,
        class_weights=class_weights,
        use_pixel_aux=args.use_pixel_aux,
        lambda_pixel=args.lambda_pixel,
        pixel_fusion_weight=args.pixel_fusion_weight,
        use_earth_adapter=True,
        earth_dim=args.earth_dim,
        earth_adapter_start=args.earth_adapter_start,
        earth_fft_start=args.earth_fft_start,
        earth_cutoff_ratio=args.earth_cutoff_ratio,
        earth_scale=args.earth_scale,
        use_structure_injection=cfg["structure"],
        structure_scale=args.structure_scale,
    ).to(device)

    tcsb = None
    if cfg["rescue"]:
        tcsb = FrozenTCSB(
            sam_ckpt=args.sam_ckpt,
            lora_ckpt=args.tcsb_lora_ckpt,
            device=device,
        )

    start_epoch = 1
    resume_checkpoint = None
    if args.resume_ckpt:
        checkpoint = torch.load(args.resume_ckpt, map_location="cpu")
        if checkpoint.get("variant") not in (None, args.variant):
            raise ValueError(
                f"Resume variant {checkpoint.get('variant')} != requested {args.variant}"
            )
        missing, unexpected = model.load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        resume_checkpoint = checkpoint
        print(
            f"[Resume] epoch={start_epoch - 1}, missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    if resume_checkpoint is not None and "optimizer_state_dict" in resume_checkpoint:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        print("[Resume] optimizer state restored")

    print(f"[TSDB] Training | epochs={args.epochs} "
          f"train_images={len(train_set)} val_images={len(val_set)}", flush=True)
    print(f"[TSDB] Output: {work_dir}", flush=True)

    history = []
    history_path = work_dir / "history.json"
    if args.resume_ckpt and history_path.exists():
        with open(history_path, "r", encoding="utf-8") as handle:
            history = json.load(handle)

    last_miou = None
    last_ious = None
    for epoch in range(start_epoch, args.epochs + 1):
        train_log = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.resize,
            args,
            epoch,
            tcsb=tcsb,
        )
        log_epoch("TSDB", epoch, args.epochs, train_log)
        should_eval = epoch % args.eval_interval == 0 or epoch == args.epochs
        record = {"epoch": epoch, "train_log": train_log}
        current_miou = None
        current_ious = None
        if should_eval:
            current_miou, current_ious = base.evaluate(
                model, val_loader, device, args.resize
            )
            last_miou, last_ious = current_miou, current_ious
            record["target_monitor_mIoU"] = last_miou
            record["per_class_iou"] = {
                key: float(value) for key, value in zip(base.CLASS_NAMES, last_ious)
            }

        history.append(record)
        with open(history_path, "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

        if epoch % args.save_interval == 0 and epoch != args.epochs:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                current_miou,
                current_ious,
                args,
                work_dir / f"epoch_{epoch:02d}.pth",
                role="fixed_epoch_snapshot",
                tcsb=tcsb,
            )


    save_checkpoint(
        model,
        optimizer,
        args.epochs,
        last_miou,
        last_ious,
        args,
        work_dir / "final_tsdb.pth",
        role="formal_final_epoch",
        tcsb=tcsb,
    )
    print(f"[TSDB] Saved: {work_dir / 'final_tsdb.pth'}", flush=True)


if __name__ == "__main__":
    main()
