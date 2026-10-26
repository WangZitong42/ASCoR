import math
from typing import Iterable, Optional

import torch
import torch.nn as nn
from torch import Tensor


class RAFDRAAdapter(nn.Module):


    def __init__(
        self,
        embed_dim: int = 1024,
        dim: int = 24,
        num_layers: int = 32,
        adapter_layers: Optional[Iterable[int]] = None,
        fft_layers: Optional[Iterable[int]] = None,
        cutoff_ratio: float = 0.3,
        scale: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = int(embed_dim)
        self.dim = int(dim)
        self.num_layers = int(num_layers)
        self.cutoff_ratio = float(cutoff_ratio)

        if adapter_layers is None:
            adapter_layers = range(self.num_layers)

        if fft_layers is None:
            fft_layers = []

        self.adapter_layers = {
            int(x) for x in adapter_layers
        }

        self.fft_layers = {
            int(x) for x in fft_layers
        }

        self.scale = nn.Parameter(
            torch.full(
                (self.num_layers,),
                float(scale),
                dtype=torch.float32,
            )
        )

        self.mlp_list1 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.embed_dim, self.dim),
                    nn.ReLU(),
                    nn.Linear(self.dim, self.embed_dim),
                )
                for _ in range(self.num_layers)
            ]
        )

        if self.fft_layers:
            self.mlp_list2 = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.embed_dim, self.dim),
                        nn.ReLU(),
                        nn.Linear(self.dim, self.embed_dim),
                    )
                    for _ in range(self.num_layers)
                ]
            )

            self.mlp_list3 = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.embed_dim, self.dim),
                        nn.ReLU(),
                        nn.Linear(self.dim, self.embed_dim),
                    )
                    for _ in range(self.num_layers)
                ]
            )

            self.router = nn.ModuleList(
                [
                    nn.Linear(self.embed_dim, 3)
                    for _ in range(self.num_layers)
                ]
            )
        else:
            self.mlp_list2 = None
            self.mlp_list3 = None
            self.router = None

        self._init_weights()

    def _init_one_mlp(self, mlp):
        nn.init.kaiming_uniform_(
            mlp[0].weight,
            a=math.sqrt(5),
        )

        nn.init.kaiming_uniform_(
            mlp[2].weight,
            a=math.sqrt(5),
        )

    def _init_weights(self):
        for mlp in self.mlp_list1:
            self._init_one_mlp(mlp)

        if self.mlp_list2 is not None:
            for mlp in self.mlp_list2:
                self._init_one_mlp(mlp)

        if self.mlp_list3 is not None:
            for mlp in self.mlp_list3:
                self._init_one_mlp(mlp)

    def decompose_fft(self, feats: Tensor):


        if feats.ndim != 4:
            raise RuntimeError(
                f"RA/FDRA adapter expected 4D SAM feature, "
                f"got {tuple(feats.shape)}"
            )

        if feats.shape[-1] != self.embed_dim:
            raise RuntimeError(
                f"RA/FDRA adapter expected last dim "
                f"{self.embed_dim}, got {tuple(feats.shape)}"
            )

        original_dtype = feats.dtype


        x = (
            feats
            .permute(0, 3, 1, 2)
            .contiguous()
            .float()
        )

        fft = torch.fft.fft2(
            x,
            norm="ortho",
        )

        fft_shifted = torch.fft.fftshift(
            fft,
            dim=(-2, -1),
        )

        _, _, H, W = fft_shifted.shape

        cutoff = int(
            min(H, W)
            * self.cutoff_ratio
            // 2
        )

        cutoff = max(1, cutoff)

        cx = H // 2
        cy = W // 2

        mask_low = torch.zeros(
            (1, 1, H, W),
            device=x.device,
            dtype=x.dtype,
        )

        x1 = max(0, cx - cutoff)
        x2 = min(H, cx + cutoff)
        y1 = max(0, cy - cutoff)
        y2 = min(W, cy + cutoff)

        mask_low[
            :,
            :,
            x1:x2,
            y1:y2,
        ] = 1.0

        mask_high = 1.0 - mask_low

        fft_low = (
            fft_shifted
            * mask_low
        )

        fft_high = (
            fft_shifted
            * mask_high
        )

        low = torch.fft.ifft2(
            torch.fft.ifftshift(
                fft_low,
                dim=(-2, -1),
            ),
            norm="ortho",
        ).real

        high = torch.fft.ifft2(
            torch.fft.ifftshift(
                fft_high,
                dim=(-2, -1),
            ),
            norm="ortho",
        ).real

        low = (
            low
            .permute(0, 2, 3, 1)
            .contiguous()
            .to(original_dtype)
        )

        high = (
            high
            .permute(0, 2, 3, 1)
            .contiguous()
            .to(original_dtype)
        )

        return low, high

    def forward(
        self,
        feats: Tensor,
        layer: int,
    ) -> Tensor:

        layer = int(layer)

        if layer not in self.adapter_layers:
            return feats

        if not torch.is_tensor(feats):
            raise TypeError(
                "RA/FDRA adapter received a non-Tensor "
                f"output at SAM block {layer}: "
                f"{type(feats)}"
            )

        if feats.ndim != 4:
            raise RuntimeError(
                f"Unexpected SAM block {layer} feature: "
                f"{tuple(feats.shape)}"
            )

        if feats.shape[-1] != self.embed_dim:
            raise RuntimeError(
                f"SAM block {layer}: expected BHWC with "
                f"C={self.embed_dim}, got {tuple(feats.shape)}"
            )

        delta_original = (
            self.mlp_list1[layer](
                feats
            )
        )


        if layer not in self.fft_layers:
            return (
                feats
                + self.scale[layer]
                * delta_original
            )


        low_freq, high_freq = (
            self.decompose_fft(feats)
        )

        delta_low = (
            self.mlp_list2[layer](
                low_freq
            )
        )

        delta_high = (
            self.mlp_list3[layer](
                high_freq
            )
        )

        router_weights = (
            self.router[layer](
                feats
            )
        )

        router_weights = torch.softmax(
            router_weights,
            dim=-1,
        )

        delta_feat = (
            router_weights[..., 0:1]
            * delta_original
            +
            router_weights[..., 1:2]
            * delta_low
            +
            router_weights[..., 2:3]
            * delta_high
        )

        return (
            feats
            + self.scale[layer]
            * delta_feat
        )
