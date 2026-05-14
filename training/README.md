# TopMiner training

This folder trains the trainable parts of the paper's Section 8.1 image
detector:

- EfficientNet spatial pathway projection head
- Frequency-domain filter-bank head
- Attention gate from Eq. 11
- Final binary classifier

It is separate from the app runtime. It produces a `.pth` checkpoint that can
be wired into the local detector after it is trained and evaluated.

## Layout

```text
training/
  model.py        TopMinerImageDetector + pathways + attention gate
  data.py         Folder and HuggingFace datasets, augmentation pipeline
  train.py        Focal loss, AdamW, cosine LR, per-epoch validation
  eval.py         Standalone evaluation on a saved checkpoint
  checkpoints/    Output directory, gitignored
```

## Dataset

### Local folder

```text
my_data/
  real/
    img001.jpg
  fake/
    ai001.jpg
```

### OpenFake on HuggingFace

OpenFake is `ComplexDataLab/OpenFake`. The default config is `core`, with
`train`, `validation`, and `test` splits. Labels are strings: `real` and
`fake`.

CPU/GPU pilot run with caps:

```powershell
$env:PYTHONPATH="."
python training/train.py `
    --hf-dataset ComplexDataLab/OpenFake `
    --hf-config core `
    --hf-train-split "train[:2000]" `
    --hf-val-split "validation[:500]" `
    --hf-label-real-value real `
    --backbone efficientnet_b0 `
    --image-size 224 `
    --batch-size 8 `
    --epochs 1
```

For an offline wiring smoke test, add `--no-pretrained-backbone`. Do not use
that option for real training unless you intentionally want a randomly
initialized EfficientNet backbone.

If HuggingFace's dataset builder is too slow for OpenFake, download one parquet
shard and smoke-test from it directly:

```powershell
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='ComplexDataLab/OpenFake', repo_type='dataset', filename='core/train-00023-of-00032-00018.parquet'))"

python training/train.py `
    --parquet C:\path\to\train-00023-of-00032-00018.parquet `
    --hf-label-real-value real `
    --max-parquet-samples 16 `
    --backbone efficientnet_b0 `
    --image-size 224 `
    --batch-size 4 `
    --epochs 1 `
    --no-pretrained-backbone
```

Fuller GPU run:

```powershell
$env:PYTHONPATH="."
python training/train.py `
    --hf-dataset ComplexDataLab/OpenFake `
    --hf-config core `
    --hf-label-real-value real `
    --backbone efficientnet_b4 `
    --image-size 380 `
    --batch-size 32 `
    --epochs 20 `
    --aug-level 2
```

In-the-wild OpenFake evaluation:

```powershell
$env:PYTHONPATH="."
python training/eval.py `
    --checkpoint training/checkpoints/topminer.pth `
    --hf-dataset ComplexDataLab/OpenFake `
    --hf-config reddit `
    --hf-split test `
    --hf-label-real-value real
```

## Other HuggingFace Datasets

Any dataset with an image field and a label field can work:

```powershell
python training/train.py --hf-dataset some/dataset --hf-image-field image --hf-label-field label
```

If the real class is numeric `0`, the default is fine. If it is a string, pass
`--hf-label-real-value real` or the value used by that dataset.

## Local Folder Training

```powershell
$env:PYTHONPATH="."
python training/train.py `
    --data path\to\my_data `
    --backbone efficientnet_b0 `
    --image-size 224 `
    --batch-size 16 `
    --epochs 3 `
    --aug-level 1
```

## What Trains

- Spatial pathway projection: always trains.
- Frequency filter-bank head: always trains.
- Attention gate: always trains.
- Final classifier: always trains.
- EfficientNet backbone: frozen by default. Use `--unfreeze-backbone` only when
  you have enough data and GPU time.

## Outputs

- Best checkpoint by validation F1: `training/checkpoints/topminer.pth`
- Per-epoch metrics: `training/checkpoints/train_log.csv`
- Optional per-epoch checkpoints with `--save-every-epoch`
