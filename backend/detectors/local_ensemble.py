"""Local PikSign deepfake ensemble.

This module intentionally loads only architectures that match the cached
PikSign checkpoints by name and tensor shape. A model is included in inference
only after a strict state_dict load succeeds.

The previous implementation used plausible stand-ins and `strict=False`, which
could report success while loading zero learned weights. This file vendors the
minimal inference architecture needed for the cached PikSign weights under
backend/assets/piksign_weights.
"""

from __future__ import annotations

import math
import threading
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

_BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = _BACKEND_DIR / "assets" / "piksign_weights"


def _strip_prefix(state_dict: dict, prefix: str = "module.") -> OrderedDict:
    return OrderedDict(
        (k.replace(prefix, "", 1) if k.startswith(prefix) else k, v)
        for k, v in state_dict.items()
    )


def _load_ckpt(path: str | Path, device: str) -> dict:
    raw = torch.load(path, map_location=device, weights_only=False)
    if isinstance(raw, dict):
        for key in ("state_dict", "model", "net"):
            if key in raw and isinstance(raw[key], dict):
                return _strip_prefix(raw[key])
        first = next(iter(raw.keys()), "")
        if isinstance(first, str) and "." in first:
            return _strip_prefix(raw)
    raise RuntimeError(f"checkpoint is not a state_dict: {path}")


def _state_dict_coverage(model: nn.Module, state_dict: dict) -> dict:
    model_sd = model.state_dict()
    matched = [
        k for k, v in state_dict.items()
        if k in model_sd and hasattr(v, "shape") and tuple(v.shape) == tuple(model_sd[k].shape)
    ]
    model_params = sum(v.numel() for v in model_sd.values() if hasattr(v, "numel"))
    matched_params = sum(model_sd[k].numel() for k in matched)
    ckpt_params = sum(v.numel() for v in state_dict.values() if hasattr(v, "numel"))
    return {
        "model_params": int(model_params),
        "ckpt_params": int(ckpt_params),
        "matched_keys": len(matched),
        "model_keys": len(model_sd),
        "ckpt_keys": len(state_dict),
        "coverage": float(matched_params / model_params) if model_params else 0.0,
    }


def _find_weight(filename: str, cache_dir: Path) -> Optional[Path]:
    direct = cache_dir / filename
    if direct.exists():
        return direct
    matches = list(cache_dir.rglob(filename)) if cache_dir.exists() else []
    return matches[0] if matches else None


if HAS_TORCH:

    class _CenterCropSquare:
        def __call__(self, img: Image.Image) -> Image.Image:
            side = min(img.size)
            return transforms.CenterCrop(side)(img)


    _TF_PIKSIGN_BASE = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        _CenterCropSquare(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    _TF_UCF = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    _TF_SPSL = _TF_UCF


    def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
        return nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)


    def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
        return nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)


    class _Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None):
            super().__init__()
            self.conv1 = _conv1x1(inplanes, planes)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = _conv3x3(planes, planes, stride)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = _conv1x1(planes, planes * self.expansion)
            self.bn3 = nn.BatchNorm2d(planes * self.expansion)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            identity = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            if self.downsample is not None:
                identity = self.downsample(x)
            return self.relu(out + identity)


    class PikSignSmallResNet(nn.Module):
        """PikSign custom two-stage ResNet used by npr.pth and base.pth."""

        def __init__(self, num_classes: int = 1):
            super().__init__()
            self.inplanes = 64
            self.unfoldSize = 2
            self.unfoldIndex = 0
            self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
            self.layer1 = self._make_layer(_Bottleneck, 64, 3)
            self.layer2 = self._make_layer(_Bottleneck, 128, 4, stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc1 = nn.Linear(512, num_classes)

        def _make_layer(self, block, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    _conv1x1(self.inplanes, planes * block.expansion, stride),
                    nn.BatchNorm2d(planes * block.expansion),
                )

            layers = [block(self.inplanes, planes, stride, downsample)]
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes))
            return nn.Sequential(*layers)

        @staticmethod
        def interpolate(img, factor: float):
            return F.interpolate(
                F.interpolate(img, scale_factor=factor, mode="nearest", recompute_scale_factor=True),
                scale_factor=1 / factor,
                mode="nearest",
                recompute_scale_factor=True,
            )

        def forward(self, x):
            npr = x - self.interpolate(x, 0.5)
            x = self.conv1(npr * 2.0 / 3.0)
            x = self.relu(self.bn1(x))
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            return self.fc1(x)


    class SeparableConv2d(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
            super().__init__()
            self.conv1 = nn.Conv2d(
                in_channels, in_channels, kernel_size, stride, padding, dilation,
                groups=in_channels, bias=bias,
            )
            self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

        def forward(self, x):
            return self.pointwise(self.conv1(x))


    class Block(nn.Module):
        def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
            super().__init__()
            if out_filters != in_filters or strides != 1:
                self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
                self.skipbn = nn.BatchNorm2d(out_filters)
            else:
                self.skip = None

            self.relu = nn.ReLU(inplace=True)
            rep = []
            filters = in_filters
            if grow_first:
                rep.append(self.relu)
                rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
                rep.append(nn.BatchNorm2d(out_filters))
                filters = out_filters

            for _ in range(reps - 1):
                rep.append(self.relu)
                rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
                rep.append(nn.BatchNorm2d(filters))

            if not grow_first:
                rep.append(self.relu)
                rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
                rep.append(nn.BatchNorm2d(out_filters))

            if not start_with_relu:
                rep = rep[1:]
            else:
                rep[0] = nn.ReLU(inplace=False)

            if strides != 1:
                rep.append(nn.MaxPool2d(3, strides, 1))
            self.rep = nn.Sequential(*rep)

        def forward(self, inp):
            x = self.rep(inp)
            skip = self.skipbn(self.skip(inp)) if self.skip is not None else inp
            return x + skip


    class Xception(nn.Module):
        """DeepfakeBench Xception with matching state_dict names."""

        def __init__(self, inc: int = 3, num_classes: int = 2, mode: str = "adjust_channel", dropout=False):
            super().__init__()
            self.num_classes = num_classes
            self.mode = mode
            self.conv1 = nn.Conv2d(inc, 32, 3, 2, 0, bias=False)
            self.bn1 = nn.BatchNorm2d(32)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
            self.bn2 = nn.BatchNorm2d(64)
            self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
            self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
            self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)
            self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
            self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
            self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
            self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
            self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
            self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
            self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
            self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
            self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)
            self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
            self.bn3 = nn.BatchNorm2d(1536)
            self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
            self.bn4 = nn.BatchNorm2d(2048)

            final_channel = 512 if mode == "adjust_channel_iid" else 2048
            if mode == "adjust_channel_iid":
                self.mode = "adjust_channel"
            self.last_linear = nn.Linear(final_channel, num_classes)
            if dropout:
                self.last_linear = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(final_channel, num_classes))

            self.adjust_channel = nn.Sequential(
                nn.Conv2d(2048, 512, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=False),
            )

        def fea_part1(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            return x

        def fea_part2(self, x):
            return self.block3(self.block2(self.block1(x)))

        def fea_part3(self, x):
            if self.mode == "shallow_xception":
                return x
            return self.block7(self.block6(self.block5(self.block4(x))))

        def fea_part4(self, x):
            if self.mode == "shallow_xception":
                return self.block12(x)
            return self.block12(self.block11(self.block10(self.block9(self.block8(x)))))

        def fea_part5(self, x):
            x = self.relu(self.bn3(self.conv3(x)))
            return self.bn4(self.conv4(x))

        def features(self, input):
            x = self.fea_part5(self.fea_part4(self.fea_part3(self.fea_part2(self.fea_part1(input)))))
            if self.mode == "adjust_channel":
                x = self.adjust_channel(x)
            return x

        def classifier(self, features, id_feat=None):
            x = features if self.mode == "adjust_channel" else self.relu(features)
            if len(x.shape) == 4:
                x = F.adaptive_avg_pool2d(x, (1, 1))
                x = x.view(x.size(0), -1)
            self.last_emb = x
            return self.last_linear(x - id_feat) if id_feat is not None else self.last_linear(x)

        def forward(self, input):
            x = self.features(input)
            return self.classifier(x), x


    def _r_double_conv(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )


    class AdaIN(nn.Module):
        def __init__(self, eps=1e-5):
            super().__init__()
            self.eps = eps

        def c_norm(self, x, bs, ch, eps=1e-7):
            x_var = x.var(dim=-1) + eps
            x_std = x_var.sqrt().view(bs, ch, 1, 1)
            x_mean = x.mean(dim=-1).view(bs, ch, 1, 1)
            return x_std, x_mean

        def forward(self, x, y):
            size = x.size()
            bs, ch = size[:2]
            x_std, x_mean = self.c_norm(x.view(bs, ch, -1), bs, ch, eps=self.eps)
            y_std, y_mean = self.c_norm(y.reshape(bs, ch, -1), bs, ch, eps=self.eps)
            return ((x - x_mean.expand(size)) / x_std.expand(size)) * y_std.expand(size) + y_mean.expand(size)


    class Conditional_UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.maxpool = nn.MaxPool2d(2)
            self.dropout = nn.Dropout(p=0.3)
            self.adain3 = AdaIN()
            self.adain2 = AdaIN()
            self.adain1 = AdaIN()
            self.dconv_up3 = _r_double_conv(512, 256)
            self.dconv_up2 = _r_double_conv(256, 128)
            self.dconv_up1 = _r_double_conv(128, 64)
            self.conv_last = nn.Conv2d(64, 3, 1)
            self.up_last = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
            self.activation = nn.Tanh()

        def forward(self, c, x):
            x = self.adain3(x, c)
            x = self.dropout(self.upsample(x))
            x = self.dconv_up3(x)
            c = self.dropout(self.upsample(c))
            c = self.dconv_up3(c)
            x = self.adain2(x, c)
            x = self.dropout(self.upsample(x))
            x = self.dconv_up2(x)
            c = self.dropout(self.upsample(c))
            c = self.dconv_up2(c)
            x = self.adain1(x, c)
            x = self.dropout(self.upsample(x))
            x = self.dconv_up1(x)
            return self.activation(self.up_last(self.conv_last(x)))


    class Conv2d1x1(nn.Module):
        def __init__(self, in_f, hidden_dim, out_f):
            super().__init__()
            self.conv2d = nn.Sequential(
                nn.Conv2d(in_f, hidden_dim, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(hidden_dim, out_f, 1, 1),
            )

        def forward(self, x):
            return self.conv2d(x)


    class Head(nn.Module):
        def __init__(self, in_f, hidden_dim, out_f):
            super().__init__()
            self.do = nn.Dropout(0.2)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.mlp = nn.Sequential(
                nn.Linear(in_f, hidden_dim),
                nn.LeakyReLU(inplace=True),
                nn.Linear(hidden_dim, out_f),
            )

        def forward(self, x):
            bs = x.size(0)
            x_feat = self.pool(x).view(bs, -1)
            x = self.do(self.mlp(x_feat))
            return x, x_feat


    class UCFDetector(nn.Module):
        """DeepfakeBench UCF inference graph, without training-only losses."""

        def __init__(self, specific_task_number: int = 2):
            super().__init__()
            self.num_classes = 2
            self.encoder_feat_dim = 512
            self.half_fingerprint_dim = 256
            self.encoder_f = Xception(inc=3, num_classes=2, mode="adjust_channel", dropout=False)
            self.encoder_c = Xception(inc=3, num_classes=2, mode="adjust_channel", dropout=False)
            self.lr = nn.LeakyReLU(inplace=True)
            self.do = nn.Dropout(0.2)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.con_gan = Conditional_UNet()
            self.head_spe = Head(256, 512, specific_task_number)
            self.head_sha = Head(256, 512, 2)
            self.block_spe = Conv2d1x1(512, 256, 256)
            self.block_sha = Conv2d1x1(512, 256, 256)

        def forward(self, x):
            forgery_features = self.encoder_f.features(x)
            f_share = self.block_sha(forgery_features)
            out_sha, _ = self.head_sha(f_share)
            return out_sha


    class BmUCFDetector(UCFDetector):
        pass


    class SPSLDetector(nn.Module):
        """DeepfakeBench SPSL-style 4-channel Xception wrapper."""

        def __init__(self):
            super().__init__()
            self.backbone = Xception(inc=4, num_classes=2, mode="normal", dropout=False)

        @staticmethod
        def phase_spectrum(x):
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
            fft = torch.fft.fft2(gray)
            phase = torch.angle(torch.fft.fftshift(fft))
            return (phase + math.pi) / (2 * math.pi)

        def forward(self, x):
            phase = self.phase_spectrum(x)
            logits, _ = self.backbone(torch.cat([x, phase], dim=1))
            return logits


    class NPRDetector(PikSignSmallResNet):
        pass


    class BaseDetectorEffNet(PikSignSmallResNet):
        """Compatibility name: base.pth is the same small ResNet family."""


@dataclass(frozen=True)
class _ModelSpec:
    name: str
    filename: str
    cls_name: str
    transform_name: str
    weight: float
    output: str
    enabled: bool = True


_MODEL_SPECS: List[_ModelSpec] = [
    _ModelSpec("npr", "npr.pth", "NPRDetector", "base", 0.34, "sigmoid"),
    _ModelSpec("ucf", "bm-faces-ffhq.pth", "UCFDetector", "ucf", 0.33, "softmax_ai"),
    _ModelSpec("bm_ucf", "bm-faces-v1.pth", "BmUCFDetector", "ucf", 0.33, "softmax_ai"),
    _ModelSpec("spsl", "spsl_best.pth", "SPSLDetector", "spsl", 0.0, "softmax_ai", enabled=False),
    _ModelSpec(
        "base_effnet",
        "base.pth",
        "BaseDetectorEffNet",
        "base",
        0.0,
        "sigmoid",
        enabled=False,
    ),
    # The cached TALL checkpoint is a Swin/thumbnail-layout model. It is kept
    # disabled until the exact upstream TALL graph is vendored.
]


@dataclass
class LocalEnsembleResult:
    available: bool
    status: str
    probability: float
    provider_status: str
    model_scores: Dict[str, float] = field(default_factory=dict)
    loaded_models: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    error: Optional[str] = None
    load_audit: Dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "status": self.status,
            "probability": self.probability,
            "provider_status": self.provider_status,
            "model_scores": self.model_scores,
            "loaded_models": self.loaded_models,
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
            "load_audit": self.load_audit,
        }


class LocalEnsemblePathway:
    """Local replacement for external image-classifier pathways."""

    def __init__(
        self,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        enable_effnet_spatial: bool = False,
        verbose: bool = False,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.enable_effnet_spatial = enable_effnet_spatial
        self.verbose = verbose
        self._models: Dict[str, Any] = {}
        self._transforms: Dict[str, Any] = {}
        self._outputs: Dict[str, str] = {}
        self._weights: Dict[str, float] = {}
        self._load_audit: Dict[str, dict] = {}
        self._load_lock = threading.Lock()
        self._loaded = False
        self.error: Optional[str] = None

        if not HAS_TORCH:
            self.error = "torch / torchvision / Pillow not installed"
            self.device = "cpu"
            return
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def available(self) -> bool:
        return HAS_TORCH and self.error is None

    def detect(self, image_path: str) -> LocalEnsembleResult:
        if not self.available:
            return LocalEnsembleResult(
                available=False,
                status="unavailable",
                probability=0.0,
                provider_status="unavailable",
                error=self.error,
            )

        t0 = time.time()
        try:
            self._ensure_loaded()
        except Exception as e:
            if self.verbose:
                traceback.print_exc()
            return LocalEnsembleResult(
                available=False,
                status="error",
                probability=0.0,
                provider_status="load_failed",
                error=f"model load failed: {e}",
                elapsed_seconds=time.time() - t0,
                load_audit=self._load_audit,
            )

        if not self._models:
            return LocalEnsembleResult(
                available=False,
                status="error",
                probability=0.0,
                provider_status="no_models",
                error="no strict-loaded local PikSign models are available",
                elapsed_seconds=time.time() - t0,
                load_audit=self._load_audit,
            )

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            return LocalEnsembleResult(
                available=True,
                status="error",
                probability=0.0,
                provider_status="image_open_failed",
                error=str(e),
                elapsed_seconds=time.time() - t0,
                loaded_models=list(self._models),
                load_audit=self._load_audit,
            )

        scores: Dict[str, float] = {}
        with torch.no_grad():
            for name, model in self._models.items():
                try:
                    inp = self._transforms[name](img).unsqueeze(0).to(self.device)
                    logits = model(inp)
                    if self._outputs[name] == "softmax_ai":
                        score = torch.softmax(logits, dim=1)[:, 1].item()
                    else:
                        score = torch.sigmoid(logits).flatten()[0].item()
                    scores[name] = float(min(max(score, 0.0), 1.0))
                except Exception:
                    if self.verbose:
                        traceback.print_exc()

        if not scores:
            return LocalEnsembleResult(
                available=True,
                status="error",
                probability=0.0,
                provider_status="all_failed",
                error="all strict-loaded models failed at inference",
                loaded_models=list(self._models),
                elapsed_seconds=time.time() - t0,
                load_audit=self._load_audit,
            )

        active_w = {k: self._weights[k] for k in scores}
        total = sum(active_w.values()) or 1.0
        prob = sum((active_w[k] / total) * scores[k] for k in scores)
        prob = float(min(max(prob, 0.0), 1.0))

        return LocalEnsembleResult(
            available=True,
            status="success",
            probability=prob,
            provider_status="AI" if prob >= 0.5 else "AUTHENTIC",
            model_scores={k: round(v, 4) for k, v in scores.items()},
            loaded_models=list(self._models),
            elapsed_seconds=time.time() - t0,
            load_audit=self._load_audit,
        )

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return

            class_map = {
                "NPRDetector": NPRDetector,
                "UCFDetector": UCFDetector,
                "BmUCFDetector": BmUCFDetector,
                "SPSLDetector": SPSLDetector,
                "BaseDetectorEffNet": BaseDetectorEffNet,
            }
            transform_map = {
                "base": _TF_PIKSIGN_BASE,
                "ucf": _TF_UCF,
                "spsl": _TF_SPSL,
            }

            for spec in _MODEL_SPECS:
                if not spec.enabled:
                    continue
                try:
                    wpath = _find_weight(spec.filename, self.cache_dir)
                    if not wpath:
                        self._load_audit[spec.name] = {"status": "missing", "filename": spec.filename}
                        continue

                    model = class_map[spec.cls_name]().to(self.device)
                    sd = _load_ckpt(wpath, self.device)
                    audit = _state_dict_coverage(model, sd)
                    audit.update({"status": "checked", "path": str(wpath)})
                    self._load_audit[spec.name] = audit

                    if audit["coverage"] < 0.999:
                        audit["status"] = "rejected_coverage"
                        continue

                    model.load_state_dict(sd, strict=True)
                    model.eval()
                    self._models[spec.name] = model
                    self._transforms[spec.name] = transform_map[spec.transform_name]
                    self._outputs[spec.name] = spec.output
                    self._weights[spec.name] = spec.weight
                    audit["status"] = "loaded_strict"
                except Exception as e:
                    self._load_audit[spec.name] = {
                        "status": "error",
                        "error": str(e),
                    }
                    if self.verbose:
                        traceback.print_exc()

            self._loaded = True


def detect_deepfake(image_path: str, device: Optional[str] = None) -> dict:
    detector = LocalEnsemblePathway(device=device)
    result = detector.detect(image_path)
    data = result.to_dict()
    data["is_fake"] = result.provider_status == "AI"
    return data
