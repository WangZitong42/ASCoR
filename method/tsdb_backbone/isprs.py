import argparse
import json
import math
import os
import random
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from method.logging_utils import progress as tqdm

import torch
from method.tsdb_backbone.adapters import RAFDRAAdapter
from mmengine.config import ConfigDict
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from mmseg.models.decode_heads.mask2former_head import Mask2FormerHead
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData


CLASS_NAMES = [
    "impervious_surface",
    "building",
    "low_vegetation",
    "tree",
    "car",
    "clutter_background",
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

    out = np.full(raw_mask.shape, IGNORE_INDEX, dtype=np.uint8)
    valid = (raw_mask >= 0) & (raw_mask < NUM_CLASSES)
    out[valid] = raw_mask[valid].astype(np.uint8)
    out[raw_mask == IGNORE_INDEX] = IGNORE_INDEX
    return out


class ISPRSSingleImageDataset(Dataset):
    def __init__(self, img_dir: str, ann_dir: str, resize: int = 512, max_items: Optional[int] = None):
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.resize = int(resize)
        self.files = sorted(self.img_dir.glob("*.png"))
        self.files = [p for p in self.files if (self.ann_dir / p.name).exists()]
        if max_items is not None:
            self.files = self.files[: int(max_items)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict:
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


def unwrap_processor_for_training(processor):

    if hasattr(processor.set_image, "__wrapped__"):
        processor.set_image = processor.set_image.__wrapped__.__get__(processor, processor.__class__)
    if hasattr(processor.set_text_prompt, "__wrapped__"):
        processor.set_text_prompt = processor.set_text_prompt.__wrapped__.__get__(processor, processor.__class__)
    if hasattr(processor, "_forward_grounding") and hasattr(processor._forward_grounding, "__wrapped__"):
        processor._forward_grounding = processor._forward_grounding.__wrapped__.__get__(processor, processor.__class__)


def make_seg_data_sample(gt: torch.Tensor, resize: int) -> SegDataSample:

    sample = SegDataSample()
    sample.set_metainfo(
        dict(
            img_shape=(resize, resize),
            ori_shape=(resize, resize),
            pad_shape=(resize, resize),
            batch_input_shape=(resize, resize),
            scale_factor=(1.0, 1.0),
        )
    )
    sample.gt_sem_seg = PixelData(data=gt.unsqueeze(0))
    return sample


def pil_to_tensor01(image_pil: Image.Image, device: torch.device):
    arr = np.asarray(image_pil).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t


def sum_loss_dict(losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
    total = None
    log_vars = {}
    for k, v in losses.items():
        if isinstance(v, (list, tuple)):
            val = sum(x.mean() for x in v)
        else:
            val = v.mean()
        log_vars[k] = float(val.detach().cpu())
        if "loss" in k:
            total = val if total is None else total + val
    if total is None:
        raise RuntimeError(f"No loss key found in loss dict keys: {list(losses.keys())}")
    return total, log_vars


class DynamicLoRALinear(nn.Module):

    def __init__(self, base: nn.Linear, max_r: int, alpha_mult: float = 2.0, dropout: float = 0.0, block_id: Optional[int] = None, init_active_r: Optional[int] = None):
        super().__init__()
        self.base = base
        self.block_id = -1 if block_id is None else int(block_id)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.max_r = int(max_r)
        self.alpha_mult = float(alpha_mult)
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False
        dev, dt = base.weight.device, base.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(self.max_r, self.in_features, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.max_r, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if init_active_r is None:
            init_active_r = self.max_r
        self.register_buffer("active_r", torch.tensor(int(init_active_r), dtype=torch.int32), persistent=False)
        self.register_buffer("global_scale", torch.tensor(1.0, dtype=torch.float32), persistent=False)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def get_active_rank(self):
        return int(self.active_r.item())

    def forward(self, x):
        out = self.base(x)
        r = self.get_active_rank()
        A = self.lora_A[:r].to(device=x.device, dtype=x.dtype)
        B = self.lora_B[:, :r].to(device=x.device, dtype=x.dtype)
        scale = self.global_scale.to(device=x.device, dtype=x.dtype)
        out = out + scale * self.alpha_mult * F.linear(F.linear(self.dropout(x), A), B)
        return out


FOREGROUND_PROMPTS = {
    1: "building",
    2: "road",
    3: "water",
    4: "barren land",
    5: "forest",
    6: "agricultural land",
}


class GNAct(nn.Module):
    def __init__(self, c, groups=32):
        super().__init__()
        g = min(groups, c)
        while c % g != 0:
            g -= 1
        self.norm = nn.GroupNorm(g, c)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(x))


class ConvGNAct(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=None, groups=1):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=groups, bias=False)
        self.na = GNAct(out_c)

    def forward(self, x):
        return self.na(self.conv(x))


class ResBlock(nn.Module):
    def __init__(self, c, mlp_ratio=4, drop=0.0):
        super().__init__()
        h = c * mlp_ratio
        self.dw = ConvGNAct(c, c, k=7, p=3, groups=c)
        self.pw1 = nn.Conv2d(c, h, 1, bias=False)
        self.na1 = GNAct(h)
        self.pw2 = nn.Conv2d(h, c, 1, bias=False)
        self.norm = nn.GroupNorm(1, c)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        y = self.dw(x)
        y = self.pw2(self.na1(self.pw1(y)))
        y = self.drop(self.norm(y))
        return x + y


class ASPP(nn.Module):
    def __init__(self, in_c, out_c=256, dilations=(1, 3, 6, 12)):
        super().__init__()
        self.branches = nn.ModuleList()
        for d in dilations:
            if d == 1:
                self.branches.append(nn.Sequential(nn.Conv2d(in_c, out_c, 1, bias=False), GNAct(out_c)))
            else:
                self.branches.append(nn.Sequential(nn.Conv2d(in_c, out_c, 3, padding=d, dilation=d, bias=False), GNAct(out_c)))
        self.global_branch = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_c, out_c, 1, bias=False), GNAct(out_c))
        self.fuse = nn.Sequential(nn.Conv2d(out_c * (len(dilations) + 1), out_c, 1, bias=False), GNAct(out_c), ResBlock(out_c))

    def forward(self, x):
        h, w = x.shape[-2:]
        outs = [b(x) for b in self.branches]
        g = self.global_branch(x)
        outs.append(F.interpolate(g, size=(h, w), mode="bilinear", align_corners=False))
        return self.fuse(torch.cat(outs, dim=1))


class GSEQueryGenerator(nn.Module):
    def __init__(self, embed_dim=256, num_queries=100, base_channels=64, num_classes=6, drop=0.0, query_mode="add"):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.query_mode = query_mode
        self.stem = nn.Sequential(ConvGNAct(3, base_channels, 7, 2, 3), ResBlock(base_channels, drop=drop))
        self.down2 = ConvGNAct(base_channels, base_channels * 2, 3, 2, 1)
        self.stage2 = nn.Sequential(ResBlock(base_channels * 2, drop=drop), ResBlock(base_channels * 2, drop=drop))
        self.down3 = ConvGNAct(base_channels * 2, base_channels * 4, 3, 2, 1)
        self.stage3 = nn.Sequential(ResBlock(base_channels * 4, drop=drop), ResBlock(base_channels * 4, drop=drop))
        self.down4 = ConvGNAct(base_channels * 4, base_channels * 4, 3, 2, 1)
        self.stage4 = nn.Sequential(ResBlock(base_channels * 4, drop=drop), ResBlock(base_channels * 4, drop=drop))
        self.aspp = ASPP(base_channels * 4, embed_dim)
        self.pool = nn.AdaptiveAvgPool2d((num_queries, 1))
        self.geo_proj = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.learnable_query = nn.Parameter(torch.randn(num_queries, embed_dim) * 0.02)
        self.class_embed = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.02)
        self.class_to_query = nn.Linear(embed_dim, embed_dim)
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, image_tensor):

        b = image_tensor.shape[0]
        x = self.stem(image_tensor)
        x = self.stage2(self.down2(x))
        x = self.stage3(self.down3(x))
        x = self.stage4(self.down4(x))
        ctx = self.aspp(x)
        q_geo = self.pool(ctx).squeeze(-1).transpose(1, 2).contiguous()
        q_geo = self.geo_proj(q_geo)
        base = self.learnable_query.unsqueeze(0).expand(b, -1, -1)


        cls = self.class_to_query(self.class_embed).mean(dim=0).view(1, 1, -1)
        if self.query_mode == "replace":
            q = q_geo + cls
        else:
            q = base + self.gamma * q_geo + cls
        return self.out_norm(q)

    def info(self):
        return {"gse_gamma": float(self.gamma.detach().cpu())}


class LearnableQueryGenerator(nn.Module):
    def __init__(self, embed_dim=256, num_queries=100):
        super().__init__()
        self.query = nn.Parameter(torch.randn(num_queries, embed_dim) * 0.02)

    def forward(self, image_tensor):
        b = image_tensor.shape[0]
        return self.query.unsqueeze(0).expand(b, -1, -1)

    def info(self):
        return {"gse_gamma": 0.0}


@MODELS.register_module(force=True)

def _to_config_dict(obj):

    if isinstance(obj, dict):
        return ConfigDict({k: _to_config_dict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_config_dict(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_config_dict(v) for v in obj)
    return obj


class SAMQueryMask2FormerHead(Mask2FormerHead):
    def __init__(self, replace_query_feat: bool = False, **kwargs):


        for _k in [
            "pixel_decoder",
            "transformer_decoder",
            "positional_encoding",
            "loss_cls",
            "loss_mask",
            "loss_dice",
            "train_cfg",
            "test_cfg",
        ]:
            if _k in kwargs and isinstance(kwargs[_k], dict):
                kwargs[_k] = _to_config_dict(kwargs[_k])

        super().__init__(**kwargs)
        feat_channels = kwargs["feat_channels"]
        if hasattr(self, "query_embed"):
            del self.query_embed
        self.replace_query_feat = replace_query_feat
        if replace_query_feat:
            if hasattr(self, "query_feat"):
                del self.query_feat
            self.querys2feat = nn.Linear(feat_channels, feat_channels)

    def forward(self, x: Tuple[List[torch.Tensor], torch.Tensor], batch_data_samples: List[SegDataSample]):
        feats, query_embed = x
        batch_img_metas = [data_sample.metainfo for data_sample in batch_data_samples]
        batch_size = len(batch_img_metas)
        if query_embed.ndim == 2:
            query_embed = query_embed.unsqueeze(0).expand(batch_size, -1, -1)

        mask_features, multi_scale_memorys = self.pixel_decoder(feats)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            mask = decoder_input.new_zeros((batch_size,) + multi_scale_memorys[i].shape[-2:], dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)

        if self.replace_query_feat:
            query_feat = self.querys2feat(query_embed)
        else:
            query_feat = self.query_feat.weight.unsqueeze(0).repeat((batch_size, 1, 1))

        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self._forward_head(query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                key_padding_mask=None,
            )
            cls_pred, mask_pred, attn_mask = self._forward_head(
                query_feat,
                mask_features,
                multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[-2:],
            )
            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)
        return cls_pred_list, mask_pred_list


class PixelAuxHead(nn.Module):

    def __init__(self, in_channels=(256, 256, 256, 256), embed_dim=256, num_classes=6, dropout=0.1):
        super().__init__()
        self.in_channels = list(in_channels)
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, embed_dim, kernel_size=1, bias=False),
                nn.GroupNorm(32, embed_dim),
                nn.ReLU(inplace=True),
            )
            for c in self.in_channels
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(self.in_channels), embed_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(embed_dim, num_classes, kernel_size=1),
        )

    def forward(self, feats):
        target_size = feats[0].shape[-2:]
        outs = []
        for feat, conv in zip(feats, self.lateral_convs):
            x = conv(feat)
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            outs.append(x)
        x = torch.cat(outs, dim=1)
        return self.fuse(x)


class FrozenSAMReinMask2Former(nn.Module):
    def __init__(
        self,
        sam_ckpt: str,
        device: torch.device,
        resize: int = 512,
        feature_block_ids: Tuple[int, int, int, int] = (7, 15, 23, 31),
        proj_channels: Tuple[int, int, int, int] = (256, 256, 256, 256),
        feat_channels: int = 256,
        out_channels: int = 256,
        num_queries: int = 100,
        decoder_layers: int = 6,
        num_heads: int = 8,
        use_gse: bool = True,
        query_mode: str = "add",
        gse_base_channels: int = 64,
        replace_query_feat: bool = False,
        no_object_weight: float = 0.1,
        class_weights: Optional[List[float]] = None,
        use_pixel_aux: bool = False,
        lambda_pixel: float = 0.4,
        pixel_fusion_weight: float = 0.5,
        use_earth_adapter: bool = False,
        earth_dim: int = 24,
        earth_adapter_start: int = 0,
        earth_fft_start: int = 24,
        earth_cutoff_ratio: float = 0.3,
        earth_scale: float = 0.1,
    ):
        super().__init__()
        self.use_pixel_aux = bool(use_pixel_aux)
        self.lambda_pixel = float(lambda_pixel)
        self.pixel_fusion_weight = float(pixel_fusion_weight)
        self.device0 = device
        self.resize = int(resize)
        self.feature_block_ids = list(feature_block_ids)
        self.proj_channels = list(proj_channels)

        self.sam_model = build_sam3_image_model(checkpoint_path=sam_ckpt, device=str(device), eval_mode=True)
        self.sam_model.to(device)
        self.sam_model.eval()
        for p in self.sam_model.parameters():
            p.requires_grad = False
        self.processor = Sam3Processor(model=self.sam_model, device=str(device))
        unwrap_processor_for_training(self.processor)


        self.use_earth_adapter = bool(use_earth_adapter)
        self.earth_dim = int(earth_dim)
        self.earth_adapter_start = int(earth_adapter_start)
        self.earth_fft_start = int(earth_fft_start)
        self.earth_cutoff_ratio = float(earth_cutoff_ratio)
        self.earth_scale_init = float(earth_scale)

        self.earth_adapter = None
        self._earth_hook_handles = []

        if self.use_earth_adapter:
            trunk = self.sam_model.backbone.vision_backbone.trunk
            num_sam_blocks = len(trunk.blocks)

            if not (
                0 <= self.earth_adapter_start < num_sam_blocks
            ):
                raise ValueError(
                    f"Invalid earth_adapter_start="
                    f"{self.earth_adapter_start}; "
                    f"SAM has {num_sam_blocks} blocks."
                )

            if not (
                self.earth_adapter_start
                <= self.earth_fft_start
                < num_sam_blocks
            ):
                raise ValueError(
                    f"Invalid earth_fft_start="
                    f"{self.earth_fft_start}; "
                    f"SAM has {num_sam_blocks} blocks."
                )

            adapter_layers = list(
                range(
                    self.earth_adapter_start,
                    num_sam_blocks,
                )
            )

            fft_layers = list(
                range(
                    self.earth_fft_start,
                    num_sam_blocks,
                )
            )

            self.earth_adapter = RAFDRAAdapter(
                embed_dim=1024,
                dim=self.earth_dim,
                num_layers=num_sam_blocks,
                adapter_layers=adapter_layers,
                fft_layers=fft_layers,
                cutoff_ratio=self.earth_cutoff_ratio,
                scale=self.earth_scale_init,
            )

            def make_earth_hook(layer_idx):
                def earth_hook(module, inputs, output):
                    return self.earth_adapter(
                        output,
                        layer_idx,
                    )
                return earth_hook

            for layer_idx in adapter_layers:
                handle = (
                    trunk.blocks[layer_idx]
                    .register_forward_hook(
                        make_earth_hook(
                            layer_idx
                        )
                    )
                )

                self._earth_hook_handles.append(
                    handle
                )


        trunk_dim = 1024
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(trunk_dim, c, 1), nn.GroupNorm(32, c), nn.ReLU(inplace=True))
            for c in self.proj_channels
        ])

        self.pixel_head = None
        if self.use_pixel_aux:
            self.pixel_head = PixelAuxHead(
                in_channels=self.proj_channels,
                embed_dim=feat_channels,
                num_classes=NUM_CLASSES,
                dropout=0.1,
            )

        if use_gse:
            self.query_generator = GSEQueryGenerator(embed_dim=feat_channels, num_queries=num_queries, base_channels=gse_base_channels, num_classes=NUM_CLASSES, query_mode=query_mode)
        else:
            self.query_generator = LearnableQueryGenerator(embed_dim=feat_channels, num_queries=num_queries)

        if class_weights is None:
            class_weights = [1.0] * NUM_CLASSES
        cls_weight = class_weights + [no_object_weight]


        pixel_decoder = dict(
            type="mmdet.MSDeformAttnPixelDecoder",
            num_outs=3,
            norm_cfg=dict(type="GN", num_groups=32),
            act_cfg=dict(type="ReLU"),
            encoder=dict(
                num_layers=6,
                layer_cfg=dict(
                    self_attn_cfg=dict(embed_dims=feat_channels, num_heads=num_heads, num_levels=3, num_points=4, im2col_step=64, dropout=0.0, batch_first=True, norm_cfg=None, init_cfg=None),
                    ffn_cfg=dict(embed_dims=feat_channels, feedforward_channels=feat_channels * 4, num_fcs=2, ffn_drop=0.0, act_cfg=dict(type="ReLU", inplace=True)),
                ),
            ),
            positional_encoding=dict(num_feats=feat_channels // 2, normalize=True),
            init_cfg=None,
        )
        transformer_decoder = dict(
            return_intermediate=True,
            num_layers=decoder_layers,
            layer_cfg=dict(
                self_attn_cfg=dict(embed_dims=feat_channels, num_heads=num_heads, attn_drop=0.0, proj_drop=0.0, dropout_layer=None, batch_first=True),
                cross_attn_cfg=dict(embed_dims=feat_channels, num_heads=num_heads, attn_drop=0.0, proj_drop=0.0, dropout_layer=None, batch_first=True),
                ffn_cfg=dict(embed_dims=feat_channels, feedforward_channels=feat_channels * 8, num_fcs=2, act_cfg=dict(type="ReLU", inplace=True), ffn_drop=0.0, dropout_layer=None, add_identity=True),
            ),
            init_cfg=None,
        )
        train_cfg = dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type="mmdet.HungarianAssigner",
                match_costs=[
                    dict(type="mmdet.ClassificationCost", weight=2.0),
                    dict(type="mmdet.CrossEntropyLossCost", weight=5.0, use_sigmoid=True),
                    dict(type="mmdet.DiceCost", weight=5.0, pred_act=True, eps=1.0),
                ],
            ),
            sampler=dict(type="mmdet.MaskPseudoSampler"),
        )
        self.decode_head = SAMQueryMask2FormerHead(
            in_channels=list(self.proj_channels),
            strides=[4, 8, 16, 32],
            feat_channels=feat_channels,
            out_channels=out_channels,
            num_classes=NUM_CLASSES,
            num_queries=num_queries,
            num_transformer_feat_level=3,
            align_corners=False,
            pixel_decoder=pixel_decoder,
            enforce_decoder_input_project=False,
            positional_encoding=dict(num_feats=feat_channels // 2, normalize=True),
            transformer_decoder=transformer_decoder,
            loss_cls=dict(type="mmdet.CrossEntropyLoss", use_sigmoid=False, loss_weight=2.0, reduction="mean", class_weight=cls_weight),
            loss_mask=dict(type="mmdet.CrossEntropyLoss", use_sigmoid=True, loss_weight=5.0, reduction="mean"),
            loss_dice=dict(type="mmdet.DiceLoss", use_sigmoid=True, activate=True, reduction="mean", naive_dice=True, eps=1.0, loss_weight=5.0),
            train_cfg=train_cfg,
            test_cfg=dict(),
            replace_query_feat=replace_query_feat,
        )

    def _extract_raw_feats(self, image_pil: Image.Image):
        trunk = self.sam_model.backbone.vision_backbone.trunk
        feats: Dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(i):
            def hook(module, inp, out):
                feats[i] = out
            return hook

        for i in self.feature_block_ids:
            handles.append(trunk.blocks[i].register_forward_hook(make_hook(i)))
        if self.use_earth_adapter:


            _ = self.processor.set_image(image_pil)
        else:
            with torch.no_grad():
                _ = self.processor.set_image(image_pil)
        for h in handles:
            h.remove()
        outs = []
        for i in self.feature_block_ids:
            x = feats[i]
            x = x.permute(0, 3, 1, 2).contiguous()


            if not self.use_earth_adapter:
                x = x.detach()

            outs.append(x)
        return outs

    def _build_features_and_query(self, image_pil: Image.Image):
        raw = self._extract_raw_feats(image_pil)

        sizes = [raw[0].shape[-2:], (36, 36), (18, 18), (9, 9)]
        feats = []
        for idx, x in enumerate(raw):
            if x.shape[-2:] != sizes[idx]:
                x = F.interpolate(x, size=sizes[idx], mode="bilinear", align_corners=False)
            feats.append(self.proj[idx](x))
        image_tensor = pil_to_tensor01(image_pil, self.device0)
        query_embed = self.query_generator(image_tensor)
        return feats, query_embed

    def loss(self, image_pil: Image.Image, gt: torch.Tensor):
        feats, query_embed = self._build_features_and_query(image_pil)
        sample = make_seg_data_sample(gt, self.resize)
        losses = self.decode_head.loss((feats, query_embed), [sample], train_cfg=self.decode_head.train_cfg)

        if self.use_pixel_aux:
            pixel_logits = self.pixel_head(feats)
            if pixel_logits.shape[-2:] != (self.resize, self.resize):
                pixel_logits = F.interpolate(
                    pixel_logits,
                    size=(self.resize, self.resize),
                    mode="bilinear",
                    align_corners=False,
                )
            target = gt.long()
            if target.ndim == 2:
                target = target.unsqueeze(0)
            loss_pixel_raw = F.cross_entropy(pixel_logits, target, ignore_index=255)
            losses["loss_pixel_aux"] = self.lambda_pixel * loss_pixel_raw

            losses["pixel_aux_raw"] = loss_pixel_raw.detach()

        return losses

    def forward_logits(self, image_pil: Image.Image):
        feats, query_embed = self._build_features_and_query(image_pil)
        sample = SegDataSample()
        sample.set_metainfo(dict(img_shape=(self.resize, self.resize), ori_shape=(self.resize, self.resize), pad_shape=(self.resize, self.resize), batch_input_shape=(self.resize, self.resize)))
        cls_pred_list, mask_pred_list = self.decode_head((feats, query_embed), [sample])
        cls_pred = cls_pred_list[-1]
        mask_pred = mask_pred_list[-1]
        cls_score = F.softmax(cls_pred, dim=-1)[..., :NUM_CLASSES]
        mask_prob = mask_pred.sigmoid()
        seg = torch.einsum("bqc,bqhw->bchw", cls_score, mask_prob)
        logits = torch.log(seg.clamp(min=1e-6))
        logits = F.interpolate(logits, size=(self.resize, self.resize), mode="bilinear", align_corners=False)
        if self.use_pixel_aux:
            pixel_logits = self.pixel_head(feats)
            if pixel_logits.shape[-2:] != logits.shape[-2:]:
                pixel_logits = F.interpolate(
                    pixel_logits,
                    size=logits.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            logits = logits + self.pixel_fusion_weight * pixel_logits

        return logits

    def info(self):
        if hasattr(self.query_generator, "info"):
            return self.query_generator.info()
        return {}


@torch.no_grad()
def evaluate(model, loader, device, resize):
    model.eval(); model.sam_model.eval()
    total_inter = np.zeros(NUM_CLASSES, dtype=np.float64)
    total_union = np.zeros(NUM_CLASSES, dtype=np.float64)
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        img_path = batch["img_path"][0]
        image = Image.open(img_path).convert("RGB").resize((resize, resize), Image.BILINEAR)
        gt = batch["gt"][0].numpy()
        logits = model.forward_logits(image)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        valid = gt != IGNORE_INDEX
        pred_v = pred[valid]
        gt_v = gt[valid]
        for c in range(NUM_CLASSES):
            inter = np.logical_and(pred_v == c, gt_v == c).sum()
            union = np.logical_or(pred_v == c, gt_v == c).sum()
            total_inter[c] += inter
            total_union[c] += union
    ious = total_inter / np.maximum(total_union, 1.0)
    return float(np.mean(ious)), ious.tolist()
