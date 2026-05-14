"""TopMiner image detector: trainable parts of Section 8.1.

Architecture:
  Image
   - EfficientNetSpatialPathway (pathway 3, Sec 8.1)
   - FrequencyFilterBank (pathway 2, Sec 8.1, Eq. 9-10)
  -> AttentionGate (Eq. 11)
  -> Classifier head -> binary logit
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyFilterBank(nn.Module):
    """Fixed kernels feeding a trainable small CNN head."""

    def __init__(self, out_dim: int = 256):
        super().__init__()
        kernels = torch.zeros(4, 1, 3, 3)
        kernels[0, 0] = torch.tensor([[-1., -1., -1.],
                                       [-1.,  8., -1.],
                                       [-1., -1., -1.]])  # high-pass
        kernels[1, 0] = torch.ones(3, 3) / 9.0  # low-pass
        kernels[2, 0] = torch.tensor([[-1., 0., 1.],
                                       [-2., 0., 2.],
                                       [-1., 0., 1.]])  # Sobel-x
        kernels[3, 0] = torch.tensor([[-1., -2., -1.],
                                       [ 0.,  0.,  0.],
                                       [ 1.,  2.,  1.]])  # Sobel-y
        self.register_buffer("kernels", kernels)

        self.head = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        filtered = F.conv2d(gray, self.kernels, padding=1)
        return self.head(filtered)


class EfficientNetSpatialPathway(nn.Module):
    """EfficientNet backbone plus a trainable projection to a fixed feature dim."""

    def __init__(
        self,
        out_dim: int = 256,
        backbone_name: str = "efficientnet_b4",
        freeze_backbone: bool = True,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        import timm
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained_backbone, num_classes=0)
        self.proj = nn.Linear(self.backbone.num_features, out_dim)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))


class AttentionGate(nn.Module):
    """alpha_i = softmax_i(W_i dot f_i + b_i), then f_final = sum alpha_i * f_i."""

    def __init__(self, n_pathways: int, dim: int):
        super().__init__()
        self.gates = nn.ModuleList([nn.Linear(dim, 1) for _ in range(n_pathways)])

    def forward(self, features: list[torch.Tensor]):
        gate_logits = torch.cat([g(f) for g, f in zip(self.gates, features)], dim=1)
        alpha = torch.softmax(gate_logits, dim=1)
        stacked = torch.stack(features, dim=1)
        fused = (alpha.unsqueeze(-1) * stacked).sum(dim=1)
        return fused, alpha


class TopMinerImageDetector(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        backbone_name: str = "efficientnet_b4",
        freeze_backbone: bool = True,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.spatial = EfficientNetSpatialPathway(
            out_dim=dim,
            backbone_name=backbone_name,
            freeze_backbone=freeze_backbone,
            pretrained_backbone=pretrained_backbone,
        )
        self.frequency = FrequencyFilterBank(out_dim=dim)
        self.attention = AttentionGate(n_pathways=2, dim=dim)
        self.classifier = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        f_s = self.spatial(x)
        f_f = self.frequency(x)
        fused, alpha = self.attention([f_s, f_f])
        logit = self.classifier(fused)
        if return_attn:
            return logit, alpha
        return logit

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]
