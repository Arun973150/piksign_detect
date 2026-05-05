# PikSign Detect

Production AI-image and manipulation detection with multi-pathway fusion, a FastAPI API, and a drag/drop forensic UI with ELA and noise-residual overlays.

## Pathways

| Pathway | Method | Weight |
|---|---|---:|
| `ensemble` | Local strict-loaded PikSign classifier ensemble (NPR, UCF, bm-UCF, SPSL, base) | 0.60 |
| `synthid` | Embedded authenticity signal | 0.17 + bonus |
| `noise_residual` | Wavelet block-wise noise inconsistency | 0.08 |
| `metadata` | EXIF, PNG text chunks, C2PA, AI software markers | 0.06 |
| `text_analysis` | OCR confidence, glyph quality, cross-region consistency | 0.02 when text exists |
| `dimension` | Aspect ratio, known generator sizes, crop hints | informational |

The ensemble runs locally; no external API keys are required. Weights are read from `backend/assets/piksign_weights/`. CUDA is used when available, otherwise CPU.

## Quick Start

```powershell
cd piksign_detect
python -m pip install -r requirements.txt
Copy-Item .env.example .env
$env:PYTHONPATH="."
python frontend/app.py
```

Open `http://localhost:8000`.

## API

```bash
curl -X POST http://localhost:8000/api/detect -F "file=@your_image.png"
curl http://localhost:8000/healthz
```

## Docker

```bash
cd piksign_detect/docker
docker compose up -d
```

## Research Notes

ELA is implemented as JPEG recompression at quality 95 with absolute residuals, following common ELA practice and the Krawetz-style pipeline surfaced in the prior research pass: [Error level analysis](https://en.wikipedia.org/wiki/Error_level_analysis), [Image forgery detection using error level analysis and deep learning](https://www.researchgate.net/publication/332561655_Image_forgery_detection_using_error_level_analysis_and_deep_learning), [An evaluation of Error Level Analysis in image forensics](https://www.semanticscholar.org/paper/An-evaluation-of-Error-Level-Analysis-in-image-Warif-Idris/deca390b7d129fa7e557d2280e6b9e555d16ab08), and [Shedding Light on ELA](https://www.fakeimagedetector.com/blog/shedding-light-ela-comprehensive-guide-error-level-analysis/).

Noise residual analysis follows the block-wise inconsistency idea from [Exposing Image Splicing with Inconsistent Local Noise Variances](https://cse.buffalo.edu/~siweilyu/papers/iccp12.pdf), with wavelet/noise-estimation context from [Forgery Detection in Digital Images by Multi-Scale Noise Estimation](https://pmc.ncbi.nlm.nih.gov/articles/PMC8321373/) and newer PRNU/noise extraction work such as [An improved PRNU noise extraction model for highly compressed image blocks with low resolutions](https://link.springer.com/article/10.1007/s11042-024-18255-3).

## Production Notes

Uploads are written to a temporary file and deleted after detection. Every pathway is wrapped so a failure degrades the fusion rather than crashing the request. The AI-image ensemble runs locally via PyTorch and only activates checkpoints that pass a full strict-load audit. TALL is intentionally disabled until its exact Swin/thumbnail architecture is vendored. SynthID is self-contained in this folder via `backend/vendor/synthid/robust_extractor.py` and `backend/assets/robust_codebook.pkl`.
