from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file as load_safetensors
from torchvision import transforms

from data.dct import DCT_base_Rec_Module
from models import AIDE as build_aide_model


IMAGE_SIZE = 256
TO_TENSOR = transforms.ToTensor()
NORMALIZE_AND_RESIZE = transforms.Compose(
    [
        transforms.Resize([IMAGE_SIZE, IMAGE_SIZE]),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def build_aide_input_from_pil(image: Image.Image, dct_module: DCT_base_Rec_Module) -> torch.Tensor:
    image = image.convert("RGB")
    image_tensor = TO_TENSOR(image)
    x_minmin, x_maxmax, x_minmin1, x_maxmax1 = dct_module(image_tensor)

    x_0 = NORMALIZE_AND_RESIZE(image_tensor)
    x_minmin = NORMALIZE_AND_RESIZE(x_minmin)
    x_maxmax = NORMALIZE_AND_RESIZE(x_maxmax)
    x_minmin1 = NORMALIZE_AND_RESIZE(x_minmin1)
    x_maxmax1 = NORMALIZE_AND_RESIZE(x_maxmax1)

    return torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0], dim=0)


def load_model(
    repo_dir: str | Path,
    device: str | None = None,
    weights_name: str = "model.safetensors",
) -> torch.nn.Module:
    repo_dir = Path(repo_dir)
    weights_path = repo_dir / weights_name
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = build_aide_model(resnet_path=None, convnext_path=None)
    state_dict = load_safetensors(str(weights_path))
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def predict_pil_images(
    model: torch.nn.Module,
    images: Iterable[Image.Image],
    device: str | None = None,
) -> List[dict]:
    device = device or next(model.parameters()).device.type
    dct_module = DCT_base_Rec_Module()
    batch = torch.stack([build_aide_input_from_pil(img, dct_module) for img in images], dim=0).to(device)
    logits = model(batch)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()

    outputs = []
    for prob in probs:
        real_prob = float(prob[0])
        fake_prob = float(prob[1])
        label = "fake" if fake_prob >= real_prob else "real"
        outputs.append(
            {
                "label": label,
                "real_probability": round(real_prob, 6),
                "fake_probability": round(fake_prob, 6),
            }
        )
    return outputs


def _load_images(paths: Iterable[str]) -> List[Image.Image]:
    return [Image.open(path).convert("RGB") for path in paths]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AIDE image detector inference.")
    parser.add_argument("--repo_dir", type=str, default=".", help="Local path to the model repository.")
    parser.add_argument("--image", type=str, nargs="+", required=True, help="One or more image paths.")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    args = parser.parse_args()

    model = load_model(args.repo_dir, device=args.device)
    images = _load_images(args.image)
    predictions = predict_pil_images(model, images, device=args.device)

    for image_path, prediction in zip(args.image, predictions):
        print(
            {
                "image": str(image_path),
                **prediction,
            }
        )


if __name__ == "__main__":
    main()

