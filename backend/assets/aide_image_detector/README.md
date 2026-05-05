---
license: mit
library_name: pytorch
tags:
  - image-classification
  - ai-generated-image-detection
  - deepfake-detection
  - computer-vision
  - pytorch
pipeline_tag: image-classification
---

# AIDE Image Detector

This repository packages the `checkpoint-19.pth` model from the local AIDE training run at `/home/meet/Aivsre_001/AIDE/output_multisource_run1/checkpoint-19.pth` as a Hugging Face model repository. The exported weights are provided both as a safe deployment artifact in `model.safetensors` and, if uploaded, as the original PyTorch training snapshot `checkpoint-19.pth`.

The model is based on **AIDE**: a hybrid AI-generated image detector that combines frequency-forensic evidence and high-level semantic cues. In this run, the detector uses:

- A fixed **30-filter SRM high-pass bank** to expose subtle forensic residuals.
- Two **ResNet-50-style frequency encoders** that process DCT-derived reconstructions.
- A frozen **OpenCLIP ConvNeXt-XXL visual trunk** for high-level semantic/image-manifold features.
- A final **MLP fusion head** that merges ConvNeXt and forensic embeddings into a binary classifier: `real` vs `fake`.

## Architecture

The forward path in [`models/AIDE.py`](./models/AIDE.py) is:

1. Start from one RGB image.
2. Build four DCT-based reconstructed views with [`data/dct.py`](./data/dct.py):
   - `x_minmin`
   - `x_maxmax`
   - `x_minmin1`
   - `x_maxmax1`
3. Build a fifth view, `x_0`, from the normalized RGB image.
4. Pass the four DCT views through the fixed SRM high-pass filters and into two ResNet branches.
5. Pass the RGB view through the frozen OpenCLIP ConvNeXt-XXL trunk.
6. Project the ConvNeXt pooled embedding from `3072 -> 256`.
7. Average the four ResNet forensic embeddings into a single `2048`-dimensional frequency representation.
8. Concatenate `[ConvNeXt_256, Forensic_2048]` into a `2304`-dimensional vector.
9. Classify with an MLP `2304 -> 1024 -> 2`.

Important implementation details taken directly from the code:

- The frequency branch uses `HPF -> ResNet(Bottleneck, [3, 4, 6, 3])`.
- The ConvNeXt branch is constructed with `open_clip.create_model_and_transforms("convnext_xxlarge", pretrained=None)` and then populated from the checkpoint weights.
- The ConvNeXt trunk is frozen in the model definition used for this checkpoint.
- Inside the model, the RGB input is remapped from ImageNet normalization to CLIP normalization before entering the ConvNeXt visual trunk.

## Input Preparation

Inference must follow the same preparation used during training/evaluation:

1. Convert the image to RGB.
2. Convert to tensor in `[0, 1]`.
3. Use `DCT_base_Rec_Module(window_size=32, stride=16, output=256, grade_N=6)` to reconstruct four frequency-ranked views.
4. Resize all five views to `256 x 256`.
5. Normalize each view with:

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

6. Stack the views in this exact order:

```python
[x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0]
```

The provided [`inference.py`](./inference.py) script reproduces this preparation pipeline.

## Checkpoint Details

- Source checkpoint: `checkpoint-19.pth`
- Exported safe weights: `model.safetensors`
- Labels:
  - `0 -> real`
  - `1 -> fake`
- Epoch: `19`
- Logged trainable parameters: `54,432,466`

From the local training log for `output_multisource_run1`:

- Epoch 19 validation/top-1 accuracy: `77.9186`
- Epoch 19 validation loss: `0.4757`
- Best validation/top-1 accuracy observed in the same run: `78.5831` at epoch `17`

## Training Context

This checkpoint came from a multi-source run configured with:

- `data_path=/home/meet/Aivsre_001/aide_data/train_multi_v1`
- `eval_data_path=/home/meet/Aivsre_001/aide_data/eval_multi_v1`
- `epochs=20`
- `batch_size=8`
- `blr=5e-4`
- `weight_decay=0.0`
- `nb_classes=2`
- `aa=rand-m9-mstd0.5-inc1`
- `smoothing=0.1`

The upstream AIDE project is introduced in the paper **"A Sanity Check for AI-generated Image Detection"** and uses a hybrid design intended to improve robustness on challenging real-world AI-image detection settings.

## Files In This Repo

- `model.safetensors`: exported model state dict for safer deployment.
- `checkpoint-19.pth`: original PyTorch training snapshot, if uploaded.
- `config.json`: architecture and label metadata.
- `model.json`: lightweight manifest for this packaged repo.
- `preprocessor_config.json`: image normalization and DCT-view preparation metadata.
- `inference.py`: local loading and prediction helper.
- `models/` and `data/`: source modules required to reconstruct the architecture.

## Usage

Clone or download the repository, then install dependencies:

```bash
pip install -r requirements.txt
```

Run local inference:

```bash
python inference.py --repo_dir . --image /path/to/image.jpg
```

Or use it programmatically:

```python
from PIL import Image

from inference import load_model, predict_pil_images

model = load_model(".")
image = Image.open("example.jpg").convert("RGB")
result = predict_pil_images(model, [image])[0]
print(result)
```

Example output:

```python
{
    "label": "fake",
    "real_probability": 0.082134,
    "fake_probability": 0.917866,
}
```

## Notes

- This repository is designed for **weight hosting and reproducible local inference**.
- The architecture is custom and is not a native `transformers` `AutoModel` implementation.
- Because the model relies on OpenCLIP ConvNeXt-XXL plus custom DCT/SRM preprocessing, users should use the provided loader and inference script.

## Credits

This packaged repository is derived from the original AIDE implementation:

- Project: https://github.com/shilinyan99/AIDE
- Paper: https://arxiv.org/abs/2406.19435

Original AIDE code is MIT licensed. See [`LICENSE`](./LICENSE).
