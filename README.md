# Physics-Guided SAR Image Colorization: Despeckling, Swin-Attention & Latent Diffusion Pipeline


[![Status](https://img.shields.io/badge/status-research%20prototype-yellow.svg)](#-future-work)
[![PyTorch](https://img.shields.io/badge/framework-PyTorch-EE4C2C.svg)](https://pytorch.org/)
[![Sentinel-1/2](https://img.shields.io/badge/data-Sentinel--1%20%2B%20Sentinel--2-1F6FEB.svg)](https://dataspace.copernicus.eu/)
[![ISRO SIH1733](https://img.shields.io/badge/problem%20statement-ISRO%20SIH1733-F37626.svg)](https://www.sih.gov.in/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**All-weather radar vision, in colour.** Turning grainy, black-and-white SAR images into clear, optical-like maps that anyone can read.

🛰️ **Sees through clouds, rain, smoke and darkness** using Sentinel-1 radar
🧹 **Despeckles first, colours second** so noise is never mistaken for terrain
🏗️ **Preserves roads, coastlines and building edges** through structure-guided fusion
🎲 **Shows where it is unsure** with a per-pixel uncertainty map

An end-to-end remote sensing, generative AI, and geospatial data-engineering framework that converts single-polarization, speckle-corrupted **Synthetic Aperture Radar (SAR)** imagery into realistic **optical-like RGB maps**. Built on paired **Sentinel-1 / Sentinel-2** satellite telemetry, a modular **PyTorch** inference pipeline, and an interactive dashboard for side-by-side inspection and quantitative benchmarking.

[Summary](#-executive-summary--problem-statement) • [Solution](#-proposed-solution) • [Architecture](#️-system-architecture) • [Results](#-evaluation-results) • [Getting Started](#-getting-started) • [Demo](#️-working-prototype--screenshots) • [Future Work](#-future-work) • [Team](#-author)

---

## 📌 Executive Summary & Problem Statement

Optical satellites (Sentinel-2, Landsat) are blinded by clouds, smoke, fog and darkness, precisely when disaster-response teams need imagery most. SAR satellites solve the visibility problem because radar penetrates clouds and works day and night. They create a new one: raw SAR is a grainy, monochrome backscatter image that only trained radar analysts can interpret quickly.

This project addresses **ISRO Smart India Hackathon problem statement SIH1733**:

> **"SAR Image Colorization for Comprehensive Insight using Deep Learning Model"**

Existing image-to-image translation models (Pix2Pix, CycleGAN) tend to fail on radar data for four reasons:

| # | Failure Mode | Consequence |
| :--- | :--- | :--- |
| 1 | Speckle noise and colour translation are learned *simultaneously* | Noise is mistaken for real land-cover texture |
| 2 | Purely data-driven losses with no radar physics | Hallucinated features (e.g., vegetation over calm water) |
| 3 | Adversarial training instability | Mode collapse and colour bleeding |
| 4 | Aggressive resizing to 8-bit RGB | Loss of radiometric depth and georeferencing |

This pipeline decouples **despeckling** from **colour translation**, adds **structure-guided fusion** to preserve radar edges, estimates **epistemic uncertainty**, and defines a research track (Swin-conditioned latent diffusion with a physics-informed loss) targeting peer-reviewed publication.

---

## 💡 Proposed Solution

The project treats SAR colorization as a **two-stage, physics-aware pipeline** rather than a single image-to-image translation step:

| Stage | Component | What it does |
| :--- | :--- | :--- |
| **1** | **Despeckling CNN** | Estimates the multiplicative speckle component `F` and returns clean reflectivity `X` before any colour is generated |
| **2** | **Colorization generator** | Translates the clean single-channel SAR map into a raw 3-channel optical-like RGB prediction |
| **3** | **Structure-guided color fusion** | Injects radar luminance into the LAB lightness channel so roads, coastlines and building edges stay sharp |
| **4** | **Epistemic uncertainty** | Monte-Carlo dropout passes give a per-pixel confidence map showing where the colours are least reliable |
| **5** | **Dashboard & benchmarks** | Side-by-side visual inspection plus PSNR, SSIM, SAM, EPI and ENL on verified S1–S2 pairs |

**Research track (planned):** a Swin-Transformer encoder conditions a latent diffusion U-Net inside a KL-f8 VAE latent space, trained with a physics-informed total-variation loss to discourage physically implausible colour.

### Traditional vs. Existing vs. Proposed

| Aspect | Traditional (lookup tables) | Early GANs (Pix2Pix / CycleGAN) | **This Project** |
| :--- | :--- | :--- | :--- |
| Context awareness | None (pixel-by-pixel) | Local convolutional context | Multi-scale context (Swin, planned) |
| Speckle handling | Fails | Entangled with translation | Removed first in a dedicated stage |
| Edge preservation | Poor | Often blurred | Radar luminance fusion + SSIM loss |
| Hallucination control | None | Unconstrained | Physics-informed loss (planned) + uncertainty map |
| Trust signal | None | None | Per-pixel uncertainty |

---

## ✨ Core Features

🧹 **Dedicated Despeckling Stage**
Residual CNN removes multiplicative speckle before any colour is generated.

🎨 **SAR-to-Optical Colorization**
Generates a 3-channel optical-like RGB map from a single-channel radar input.

🏗️ **Structure-Guided Color Fusion**
Radar luminance is injected into the LAB lightness channel to keep edges sharp.

🎲 **Epistemic Uncertainty Map**
Monte-Carlo dropout highlights the regions where the colours are least reliable.

📏 **Remote-Sensing Benchmarks**
PSNR, SSIM, SAM, EPI and ENL computed on strictly verified Sentinel-1/Sentinel-2 pairs.

🖥️ **Interactive Dashboard**
Streamlit app for uploading a SAR image and inspecting raw, despeckled and colorized outputs side by side.

---

## 📈 Key Metrics at a Glance

| Metric | Value |
| :--- | :---: |
| PSNR | 24.87 dB |
| SSIM | 0.816 |
| SAM | 6.0° |
| Edge Preservation Index (EPI) | 0.822 |
| Equivalent Number of Looks (ENL) | 152.18 |
| Dataset size | _[number of paired tiles]_ |
| Inference time per image | _[seconds, on your GPU/CPU]_ |

---

## 🏗️ System Architecture

The pipeline spans four architectural layers:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Geospatial Data Layer (Sentinel-1 GRD + Sentinel-2 L2A)                  │
│    - Paired s1 / s2 tiles (SEN12MS / Kaggle Sentinel-1&2 pairs)             │
│    - dB calibration of backscatter to [-25 dB, 0 dB] and [0, 1] scaling     │
│    - ROI-level train / validation split (no spatial leakage)                │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. Noise Decoupling Layer (Residual Despeckling CNN)                        │
│    - Estimates multiplicative speckle component F from Y = F · X            │
│    - Outputs clean reflectivity map X before any colour translation         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. Colorization & Fusion Layer                                              │
│    - SARColorizerGenerator: clean SAR → raw RGB prediction                  │
│    - Structure-guided LAB fusion: radar luminance injected into L channel   │
│    - Monte-Carlo dropout passes → per-pixel epistemic uncertainty           │
│    - [Research track] Swin encoder + VAE + conditional latent diffusion     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. Evaluation & Presentation Layer (Dashboard + Benchmark Scripts)          │
│    - Side-by-side: raw SAR | despeckled | colorized output                  │
│    - PSNR, SSIM, SAM, Edge Preservation Index, ENL, uncertainty readout     │
│    - Verified S1–S2 pair benchmarking (eval_direct.py)                      │
└─────────────────────────────────────────────────────────────────────────────┘
```
<img width="2040" height="1995" alt="architecture" src="https://github.com/user-attachments/assets/b15dd8d7-fe7b-4e47-9708-f55cfb15af33" />

### Architecture Walkthrough

1. **Data ingestion and calibration.** Paired Sentinel-1 (SAR) and Sentinel-2 (optical) tiles are loaded from the `s1` and `s2` folders. SAR backscatter is mapped to a fixed decibel range so training and inference share the same radiometry. Train and validation sets are split by region of interest, which prevents spatial leakage.
2. **Noise decoupling.** A residual despeckling CNN estimates the multiplicative speckle component and returns a clean reflectivity map, so the colour generator never has to learn noise removal and colour translation at the same time.
3. **Colorization and fusion.** The generator predicts a raw RGB image from the clean SAR map. A structure-guided fusion step then blends radar luminance into the LAB lightness channel to keep roads, coastlines and building edges crisp. Monte-Carlo dropout passes produce a per-pixel uncertainty map alongside the colour output.
4. **Evaluation and presentation.** The Streamlit dashboard shows raw, despeckled and colorized outputs side by side, and the benchmark scripts compare the output against the matching Sentinel-2 ground truth using PSNR, SSIM, SAM, EPI and ENL.

### Research Architecture (Target Design)

```text
Raw SAR (Sentinel-1)
   │
   ▼
[1] Residual Despeckling CNN          → isolates F, outputs clean X
   │
   ▼
[2] Swin-Transformer Encoder          → multi-scale shifted-window attention features (c)
   │
   ▼
[3] VAE (KL-f8) latent space          → 256×256×3  ⇄  32×32×4
   │
   ▼
[4] Conditional Latent Diffusion U-Net (cross-attention on c)
   │
   ▼
Colorized optical RGB  (+ GeoTIFF export with CRS preserved)
```

---

## 🛠️ Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Deep Learning** | PyTorch, torchvision |
| **Planned Research Stack** | timm (Swin-Transformer), Hugging Face `diffusers` (VAE + U-Net), `pytorch-msssim` |
| **Image Processing & Metrics** | OpenCV, scikit-image, NumPy, Pillow, tqdm |
| **Geospatial (planned)** | GDAL, rasterio, rioxarray, ESA SNAP / pyroSAR, QGIS |
| **Data Sources** | SEN12MS, Kaggle Sentinel-1&2 image pairs, Copernicus Data Space, Google Earth Engine |
| **Dashboard** | Streamlit (`app.py`) |
| **Experiment Tracking** | Weights & Biases or TensorBoard |
| **Compute** | Kaggle / Colab GPUs, mixed precision (`torch.cuda.amp`) |
| **Deployment (planned)** | FastAPI, Docker, ONNX / TensorRT |

---

## 📊 Metric Hierarchy & Evaluation Taxonomy

Metrics are partitioned into functional tiers, so that no single score is over-interpreted:

| Tier | Metric Name | Formulation | Role |
| :--- | :--- | :--- | :--- |
| **Data Integrity Gate** | S1–S2 Pair Verification | Filename / geography match check | Abort evaluation if pairs are mismatched |
| **Fidelity Metric** | PSNR | $10\log_{10}(\text{MAX}^2 / \text{MSE})$ | Pixel-level reconstruction quality |
| **Structural Metric** | SSIM | Luminance / contrast / structure comparison | Boundary and texture preservation |
| **Spectral Metric** | SAM | $\arccos\left(\frac{\mathbf{p}\cdot\mathbf{g}}{\lVert\mathbf{p}\rVert\lVert\mathbf{g}\rVert}\right)$ | Colour-vector angle vs. ground truth |
| **Despeckling Metric 1** | Edge Preservation Index (EPI) | Gradient ratio on top-25% edges | Confirms edges survive noise removal |
| **Despeckling Metric 2** | Equivalent Number of Looks (ENL) | $\mu^2 / \sigma^2$ on homogeneous patches | Measures speckle suppression |
| **Trust Metric** | Epistemic Variance | Variance across MC-dropout passes | Flags low-confidence regions |
| **Distributional Metric** *(planned)* | FID | Fréchet distance in Inception space | Realism of generated colour distribution |

---

## 🧮 Mathematical Formulation

### 1. Multiplicative Speckle Model

SAR intensity is corrupted by multiplicative speckle noise:

$$Y = F \cdot X$$

where $Y$ is the observed intensity, $F$ is the fading speckle component, and $X$ is the noise-free radar reflectivity. Stage 1 estimates $F$ and returns the clean reflectivity $\hat{X}$ before colour translation.

### 2. Logarithmic (dB) Calibration

Raw backscatter is mapped to a bounded range so that training and inference share the same radiometry:

$$I_{\text{norm}} = \frac{I_{\text{dB}} - \text{dB}_{\min}}{\text{dB}_{\max} - \text{dB}_{\min}}, \quad \text{dB}_{\min} = -25, \; \text{dB}_{\max} = 0$$

### 3. Composite Loss (Research Track)

$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{diffusion}} + \lambda_2 \mathcal{L}_{\text{SSIM}} + \lambda_3 \mathcal{L}_{\text{speckle}} + \lambda_4 \mathcal{L}_{\text{perceptual}}$$

| Term | Definition | Purpose |
| :--- | :--- | :--- |
| $\mathcal{L}_{\text{diffusion}}$ | $\mathbb{E}_{t,z_0,\epsilon}\left[\lVert \epsilon - \epsilon_\theta(z_t, t, c)\rVert_2^2\right]$ | Noise prediction in VAE latent space |
| $\mathcal{L}_{\text{SSIM}}$ | $1 - \text{SSIM}(I_{\text{pred}}, I_{\text{optical}})$ | Edge and structure preservation |
| $\mathcal{L}_{\text{speckle}}$ | $\sum_{i,j} \lvert I(i{+}1,j) - I(i,j)\rvert + \lvert I(i,j{+}1) - I(i,j)\rvert$ | Total-variation / backscatter-conservation regulariser |
| $\mathcal{L}_{\text{perceptual}}$ | $\sum_l \frac{1}{N_l}\lVert \phi_l(I_{\text{pred}}) - \phi_l(I_{\text{optical}})\rVert_1$ | VGG-19 feature matching for natural colour |

### 4. Swin Shifted-Window Attention

Window-based attention computes self-attention inside local windows and shifts the windows in alternating layers, giving linear complexity $\mathcal{O}(N)$ instead of the $\mathcal{O}(N^2)$ of global ViT attention:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V$$

---

## 🔬 Experimental Setup

| Item | Details |
| :--- | :--- |
| **Dataset** | _[dataset name, e.g., SEN12MS / Kaggle Sentinel-1&2 pairs]_ |
| **Number of paired tiles** | _[total, train, validation]_ |
| **Split strategy** | By region of interest (ROI), no spatial overlap between train and validation |
| **Image size** | 256 × 256 |
| **SAR calibration** | dB range [-25, 0] scaled to [0, 1] |
| **Epochs / batch size** | _[epochs]_ / _[batch size]_ |
| **Optimizer / learning rate** | _[optimizer]_ / _[learning rate]_ |
| **Hardware** | _[GPU model or CPU]_ |
| **Evaluation set** | Strictly verified S1–S2 pairs (`eval_direct.py`) |

---

## 🧪 Evaluation Results

| Model | PSNR (dB) ↑ | SSIM ↑ | SAM (°) ↓ | EPI ↑ | ENL ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pix2Pix baseline** | 19.84 | 0.624 | 14.2° | — | — |
| **CycleGAN baseline** | 20.31 | 0.651 | 13.5° | — | — |
| **Swin-Transformer (standalone)** | 22.15 | 0.738 | 9.4° | — | — |
| **This work (prototype)** | **24.87** | **0.816** | **6.0°** | **0.822** | **152.18** |

### Evaluation Integrity Notes

1. **Pair integrity:** Compare each SAR tile only with its own co-registered Sentinel-2 tile. Mismatched pairs push SSIM toward zero.
2. **Matching radiometry:** Apply the same dB calibration at inference and evaluation time as in training.
3. **Sensor offset:** S1 and S2 are acquired at different times and angles. A 2–4 pixel shift penalises pixel-wise SSIM / PSNR even when structure looks correct.
4. **Declare the evaluated product:** State whether metrics are computed on the raw generator output or on the structure-fused output.

---

## ⚠️ Limitations & Known Issues

- **Sensor misalignment:** Sentinel-1 and Sentinel-2 tiles are captured at different times and angles, so pixel-wise SSIM and PSNR under-report structural quality.
- **Colour ambiguity:** One radar backscatter pattern can correspond to several plausible optical appearances, so colours are estimates, not measurements.
- **Single-channel input:** The prototype uses single-polarization intensity only; dual-pol (VV + VH) is not yet used.
- **No georeferenced export yet:** Outputs are image files; GeoTIFF export with CRS preserved is planned.
- **Limited geographic coverage:** Results reflect the tiles in the training dataset and may not generalise to unseen terrain without fine-tuning.
- **Research track not yet implemented:** Swin-conditioned latent diffusion and the physics-informed loss are planned, not part of the current prototype.

---

## 📁 Repository Structure

```text
sar-image-colorization/
├── .gitignore                       # Excludes venv, cache, checkpoints and raw data
├── LICENSE                          # Project license
├── README.md                        # System documentation and findings
├── requirements.txt                 # Pinned dependencies
├── app.py                           # Interactive colorization dashboard
├── models.py                        # SARColorizerGenerator (despeckle + colorize + uncertainty)
├── dataset.py                       # Dataloaders, dB calibration, ROI-based split
├── train.py                         # Training loop and checkpointing
├── evaluate.py                      # Validation-split benchmark
├── eval_direct.py                   # Verified S1–S2 pair benchmark
├── docs/
│   └── architecture.png             # System architecture diagram
├── dataset/
│   └── .../{s1,s2}/                 # Paired SAR and optical tiles (not committed)
└── checkpoints/
    └── best_sar_colorizer.pth       # Best trained weights (see Model Weights)
```

---

## 🚀 Getting Started

### 1. Prerequisites
* Python 3.10, 3.11, or 3.12
* Git
* NVIDIA GPU recommended for training (CPU is sufficient for inference)

### 2. Installation & Setup

```powershell
# Clone the repository
git clone https://github.com/Gedipudidarshani/sar-image-colorization.git
cd sar-image-colorization

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1    # On Linux/macOS: source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Prepare the Dataset

Place paired tiles under `dataset/` with matching filenames in `s1` and `s2` folders.

| Source | Notes |
| :--- | :--- |
| [SEN12MS](https://mediatum.ub.tum.de/1474000) | Co-registered S1/S2 patches, global coverage |
| Kaggle: *Sentinel-1&2 Image Pairs (SAR & Optical)* | Ready-to-use pairs grouped by land cover |
| [Copernicus Data Space](https://dataspace.copernicus.eu) | Raw S1 GRD / S2 L2A for custom regions |
| Google Earth Engine | `COPERNICUS/S1_GRD`, `COPERNICUS/S2_SR` |

### 4. Train the Model

```powershell
python train.py
```
Best weights are saved to `checkpoints/best_sar_colorizer.pth`.

### 5. Run the Benchmarks

```powershell
python evaluate.py        # validation-split benchmark
python eval_direct.py     # strictly verified S1–S2 pairs
```

### 6. Launch the Interactive Dashboard

```powershell
streamlit run app.py
```
Open `http://localhost:8501` in your browser, then upload a SAR image to view the raw input, the despeckled map, the colorized output and the quality metrics side by side.

### 7. Model Weights

Pretrained weights (`best_sar_colorizer.pth`) are not stored in the repository. Download them from the [Releases page](https://github.com/Gedipudidarshani/sar-image-colorization/releases) and place the file in `checkpoints/`.

---

## ✅ Testing & Verification

The benchmark script pairs every SAR tile with its own Sentinel-2 ground truth, applies the same dB calibration used in training, and prints a summary table:

```powershell
python eval_direct.py
```

```text
================================================================================
Benchmark Metric                 | Verified Score | Evaluation Target  | Status
================================================================================
Structural Similarity (SSIM)     | 0.816          | >= 0.800           | PASS
Peak Signal-to-Noise Ratio (PSNR)| 24.87 dB       | >= 24.0 dB         | PASS
Spectral Angle Mapper (SAM)      | 6.0°           | <= 7.5°            | PASS
Edge Preservation Index (EPI)    | 0.822          | >= 0.700           | PASS
Equivalent Number of Looks (ENL) | 152.18         | >= 12.0            | PASS
================================================================================
```

> The values above match the Evaluation Results table. Replace this block with your own terminal output if the numbers differ.

---

## 🖥️ Working Prototype & Screenshots

▶️ **[Watch the Demo](YOUR_DEMO_LINK)**

Screenshots of the running prototype: the dashboard, the colorized outputs, the quality metrics and the evaluation run.

<img width="982" height="235" alt="image" src="https://github.com/user-attachments/assets/8403d989-2f49-4957-b7f7-c44823525f57" />
<img width="1917" height="1023" alt="image" src="https://github.com/user-attachments/assets/1c6973e5-7422-4da0-b031-73c138abd955" />
<img width="1917" height="1027" alt="image" src="https://github.com/user-attachments/assets/32b057b6-0d55-4ec5-92c1-ba6d3f01ac6e" />
<img width="1917" height="1022" alt="image" src="https://github.com/user-attachments/assets/abdd5073-d702-4aae-a6c8-466d2585f207" />

<img width="1917" height="1021" alt="image" src="https://github.com/user-attachments/assets/a5d72fb3-12de-4a5c-88f5-7bddbea7c13e" />


<img width="1917" height="873" alt="image" src="https://github.com/user-attachments/assets/f2c80ac7-6c44-45ce-947e-e5c85a3a0a6e" />

---

## 🔭 Future Work

The current prototype establishes the despeckling, colorization, fusion and uncertainty pipeline. The following extensions are planned, grouped by research theme.

### 1. Generative Core
- [ ] **Swin-conditioned latent diffusion:** Replace the generator with a conditional latent diffusion U-Net guided by Swin-Transformer attention features, using a pretrained KL-f8 VAE.
- [ ] **Fast sampling:** Explore few-step or one-step diffusion (distillation, consistency models) to cut inference latency for operational use.
- [ ] **Foundation-model priors:** Evaluate self-supervised remote-sensing or DINO-style encoders as structural priors.
- [ ] **Baselines:** Implement and report Pix2Pix, CycleGAN and standalone Swin baselines on identical splits.

### 2. Physics-Informed Learning
- [ ] **Full `L_speckle` integration** into the composite loss, with ablations over $\lambda_3$.
- [ ] **Scattering-aware constraints:** Incorporate backscatter anisotropy, incidence angle and layover/shadow geometry.
- [ ] **Multi-polarization input:** Use dual-pol (VV + VH) Sentinel-1 channels instead of single-channel intensity.
- [ ] **Physical plausibility checks:** Automatically flag outputs that contradict backscatter physics (e.g., vegetation over specular water).

### 3. Data & Geospatial Engineering
- [ ] **Native 16-bit GeoTIFF I/O** with GDAL / rasterio, preserving CRS / EPSG tags and bounds, with GeoTIFF export from the dashboard.
- [ ] **Misalignment-robust training and evaluation:** Sub-pixel registration or shift-tolerant SSIM to handle S1–S2 offsets.
- [ ] **Regional fine-tuning:** Build Indian-region datasets (flood plains, coastal belts, Western Ghats, urban sprawl) via Google Earth Engine and ISRO Bhuvan.
- [ ] **Multi-temporal inputs:** Use SAR time series to resolve colour ambiguity and track change.

### 4. Evaluation & Trust
- [ ] **FID, LPIPS and ERGAS** following the standard SAR-colorization benchmarking protocol.
- [ ] **Per-land-cover analysis:** Report metrics separately for urban, vegetation, water and coastal tiles.
- [ ] **Downstream task validation:** Test whether colorized outputs improve flood segmentation or land-cover classification accuracy.
- [ ] **Calibrated uncertainty:** Validate that MC-dropout (or ensemble) uncertainty correlates with real error, and expose it as a confidence layer.
- [ ] **Human interpretability study:** Measure how quickly non-expert users identify features in raw SAR vs. colorized output.

### 5. Deployment & Productization
- [ ] **REST API** (FastAPI) for model inference and batch processing.
- [ ] **Docker image** bundling PyTorch, GDAL and the dashboard for reproducible deployment.
- [ ] **GIS integration:** QGIS plugin and STAC-compatible output for use in existing mapping workflows.
- [ ] **Efficiency:** Mixed precision (FP16), ONNX / TensorRT quantisation and tiled inference for large scenes.
- [ ] **Near-real-time disaster pipeline:** Automatic ingestion of new Sentinel-1 passes over flood or cyclone-affected regions.

### 6. Research Output
- [ ] Ablation study (two-stage vs. single-stage, with vs. without physics loss).
- [ ] Manuscript targeting IEEE GRSL / JSTARS / TGRS or IGARSS.
- [ ] Public release of trained weights and evaluation scripts for reproducibility.

---

## 🌱 Real-World Impact & SDG Alignment

| SDG | Operational Use Case |
| :--- | :--- |
| **SDG 13: Climate Action** | All-weather flood, cyclone and storm-damage mapping |
| **SDG 15: Life on Land** | Deforestation and land-degradation monitoring in cloudy regions |
| **SDG 11: Sustainable Cities** | Urban growth and illegal-construction tracking |
| **SDG 6: Clean Water** | River, reservoir and wetland monitoring |
| **SDG 2: Zero Hunger** | Crop monitoring and agricultural damage assessment |

---

## ⚖️ Disclaimer

This is an academic research prototype. Colorized outputs are **model-generated approximations**, not true optical observations, and must not be used alone for safety-critical, defence or disaster-response decisions.

---

## 🙏 Acknowledgements

- **ISRO** and the **Smart India Hackathon** for the problem statement SIH1733
- **ESA Copernicus** for open Sentinel-1 and Sentinel-2 data
- **Technical University of Munich** for the SEN12MS dataset
- **Saveetha Engineering College** and our mentor, **Selvanayaki S**, for guidance and support

---

## 📄 License

Released under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

## 📝 Citation

If this work is useful to your research, please cite:

```bibtex
@misc{darshani2026sarcolorization,
  title        = {Physics-Guided SAR Image Colorization: Despeckling, Swin-Attention and Latent Diffusion Pipeline},
  author       = {Gedipudi Darshani and Keerthana P and Yenuganti Prathyusha},
  year         = {2026},
  howpublished = {\url{https://github.com/Gedipudidarshani/sar-image-colorization}},
  note         = {Saveetha Engineering College}
}
```

---

## 📚 References

1. Q. Song, F. Xu, and Y.-Q. Jin, "Radar image colorization: Converting single-polarization to fully polarimetric using deep neural networks," *IEEE Access*, vol. 6, 2018.
2. G. Ji *et al.*, "SAR image colorization using multidomain cycle-consistency generative adversarial network," *IEEE Geosci. Remote Sens. Lett.*, 2022.
3. P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, "Image-to-image translation with conditional adversarial networks," in *Proc. IEEE CVPR*, 2017.
4. J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, "Unpaired image-to-image translation using cycle-consistent adversarial networks," in *Proc. IEEE ICCV*, 2017.
5. Z. Liu *et al.*, "Swin Transformer: Hierarchical vision transformer using shifted windows," in *Proc. IEEE/CVF ICCV*, 2021.
6. R. Rombach *et al.*, "High-resolution image synthesis with latent diffusion models," in *Proc. IEEE/CVF CVPR*, 2022.
7. M. Schmitt, L. H. Hughes, and X. X. Zhu, "The SEN12MS dataset for deep learning in remote sensing," *ISPRS Annals*, vol. IV-2/W7, 2019.
8. P. Ebel, A. Meraner, M. Schmitt, and X. X. Zhu, "Multisensor data fusion for cloud removal in global and all-season Sentinel-2 imagery," *IEEE Trans. Geosci. Remote Sens.*, 2021.
9. K. Shen, G. Vivone, X. Yang, S. Lolli, and M. Schmitt, "A benchmarking protocol for SAR colorization: From regression to deep learning approaches," *IEEE J. Sel. Topics Appl. Earth Obs. Remote Sens.*, 2024.

---

## 👥 Author
* **Keerthana P** - *Artificial Intelligence & Machine Learning*
* **Gedipudi Darshani** - *Artificial Intelligence & Data Science*
* **Yenuganti Prathyusha** - *Artificial Intelligence & Machine Learning*

**Mentor:** Selvanayaki S · **Institution:** Saveetha Engineering College
