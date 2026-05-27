# Technical Documentation — Film Damage Simulator & Restoration Model

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Film Damage Generation Pipeline](#2-film-damage-generation-pipeline)
3. [Restoration Model](#3-restoration-model)
4. [Training Process](#4-training-process)
5. [Inference](#5-inference)
6. [Design Decisions and Alternatives](#6-design-decisions-and-alternatives)

---

## 1. Project Overview

This project has two components:

1. **Damage Generator** — a statistical model that simulates analogue film damage by sampling real extracted artifacts from annotated scans and placing them on clean images using spatially correlated distributions.

2. **Restoration Model** — a deep learning model trained on generated (damaged, original) image pairs, which learns to reverse the damage and restore images to their clean state.

The damage generator produces training data. The restoration model learns from that data.

---

## 2. Film Damage Generation Pipeline

### 2.1 Real Scan Data Loading (`scans.py`)

The pipeline optionally starts by loading annotated real film scans. Each scan consists of:
- A `.jpg` image of a physical film scan
- A `.json` file containing manually annotated artifact regions

Each JSON entry contains:
- `points` — a list of `{x, y}` coordinates defining the artifact contour
- `label.name` — the artifact type (`Dust`, `Dirt`, `Scratch`, `Long hair`, `Short hair`)

These contours are converted to OpenCV format and used to extract the actual pixel content (the artifact image crop) and its metadata: area, bounding box, centroid, quadrant.

**Scans 8, 9, 10** have a hard-coded `1.5×` coordinate correction applied because they were annotated at a different resolution than the actual scan size.

### 2.2 Artifact Types

Five types of damage are modelled:

| Type | Description | Synthetic sources |
|------|-------------|-------------------|
| `dust` | Small dark specks | `sprinkles/` |
| `dirt` | Irregular stains, smudges | `stain/`, `spots/`, `dirt/`, `dots/`, `scratches/`, `smut/` |
| `short hair` | Short fibres | `lint/`, `hair-short/` |
| `long hair` | Long strands crossing the frame | `hair/` |
| `scratch` | Thin linear marks | real scans only by default |

### 2.3 Statistical Sampling (`sample.py`)

Rather than placing a fixed number of artifacts, the model fits a **Gamma distribution** to the observed artifact counts and sizes from real scans.

**Why Gamma distribution?**
Artifact counts and sizes are non-negative and right-skewed (most frames have few artifacts, some have many). The Gamma distribution is a natural fit for this kind of data — it models positive, skewed quantities.

Two separate distributions are fitted per artifact type:
- **Count distribution** — fitted to per-quadrant artifact counts across all scans. At generation time, a sample is drawn to decide how many artifacts of each type to place.
- **Size distribution** — fitted to artifact contour areas. At generation time, target sizes are sampled to decide how large each artifact should be.

After sampling target sizes, `sample_closest_in_area()` selects real artifact crops whose actual area is closest to the target, introducing variety while staying statistically consistent with real data.

The `--uniform` flag skips this entirely and draws a random integer in `[min_artifacts, max_artifacts]`, placing randomly selected artifacts without size fitting.

### 2.4 Spatial Placement — Perlin Noise (`generate_masks.py`)

Artifacts on real film are not uniformly distributed. Dust clusters together, scratches run in specific directions, and damage concentrates in certain regions. To replicate this, artifact positions are sampled from a **Perlin noise probability map**.

**Perlin noise** is a gradient noise function that produces smooth, spatially correlated values. When used as a probability distribution, it creates clusters of high probability (where more artifacts land) and sparse regions, matching the clustered nature of real film damage.

The process:
1. Generate a 2D Perlin noise field of the same size as the target mask
2. Normalise to `[0, 1]`
3. Use as a probability map — sample `N` positions weighted by this map (`random_perlin_with_numpy`)
4. Place each artifact centred at its sampled position

### 2.5 Artifact Rescaling and Rotation

Each artifact crop is:
1. **Rescaled** — a global rescale factor (`target_size / 2560`) adjusts for the target resolution relative to the reference 2560 px scan resolution. An additional per-artifact scale factor adjusts for the sampled target size vs. the artifact's actual area. Combined: `new_scale = global_scale × sqrt(target_area / actual_area)`.
2. **Rotated** — a random angle in `[0°, 360°]` is applied, giving variation in artifact orientation.

If an artifact placement would go outside the canvas bounds, it is clipped.

### 2.6 Procedural Line Scratches (`line_scratch()`)

50% of generated masks also receive extra procedural scratches, drawn using Perlin noise and Gaussian noise blended together:
1. Generate a Perlin noise patch
2. Take a slice and contrast-enhance it to create a faded texture
3. Draw a thin line through it
4. Subtract the line from the texture (creating a bright streak with texture)
5. Apply Gaussian blur to soften edges
6. Compress horizontally by 50%

This produces scratches that look physically plausible — they have natural texture variation and soft edges, unlike a simple straight line.

### 2.7 Mask Convention

The final mask uses an **inverted convention**:
- `255` (white) = clean background
- `0` (black) = damaged area

This is done via `np.invert()` at the end of mask generation. The apply_damage step uses this convention.

### 2.8 Applying Damage (`apply_damage.py`)

Damage is composited onto a clean image using:

```
damaged = clean + strength × 255 × (1 − mask_normalised)
```

Where `mask_normalised` is the mask divided by 255, so it ranges `[0, 1]`.

- Where `mask = 255` (background): `damaged = clean + 0` → unchanged
- Where `mask = 0` (artifact): `damaged = clean + strength × 255` → brightened

This models the physical reality of film damage — scratches and dust on film scatter or transmit more light than the surrounding film base, appearing as **bright overexposed marks** on scans.

The `strength` parameter (default `0.6`) controls how severe the damage looks. `0.0` = invisible, `1.0` = artifact areas blown out to white.

### 2.9 Synthetic-Only Mode (`--only-synthetic`)

When real scan data is unavailable, `--only-synthetic` loads artifact images from the `../synthetic/` directory instead. Each synthetic artifact is a transparent PNG; the alpha channel is thresholded to extract the artifact shape.

This mode forces `--uniform` sampling (no Gamma fitting, since there are no real scan statistics). Artifact counts are drawn uniformly at random.

---

## 3. Restoration Model

### 3.1 Architecture — U-Net

The restoration model is a **U-Net**, an encoder-decoder convolutional neural network with skip connections.

U-Net was originally proposed by Ronneberger et al. (2015) for biomedical image segmentation. It has since become standard for image-to-image translation tasks, including restoration, denoising, and inpainting.

**Structure:**

```
Input (4ch or 3ch)
    ↓
[Encoder Block 1: 64 filters]  ──────────────────────────────┐ skip
    ↓ MaxPool                                                 │
[Encoder Block 2: 128 filters] ─────────────────────────┐ skip │
    ↓ MaxPool                                            │     │
[Encoder Block 3: 256 filters] ────────────────────┐ skip │     │
    ↓ MaxPool                                       │     │     │
[Encoder Block 4: 512 filters] ──────────────┐ skip │     │     │
    ↓ MaxPool                                │     │     │     │
[Bottleneck: 1024 filters]                   │     │     │     │
    ↓ ConvTranspose (upsample)               │     │     │     │
[Decoder Block 4: 512] ←──────────────────── ┘     │     │     │
    ↓ ConvTranspose                                │     │     │
[Decoder Block 3: 256] ←───────────────────────── ┘     │     │
    ↓ ConvTranspose                                      │     │
[Decoder Block 2: 128] ←─────────────────────────────── ┘     │
    ↓ ConvTranspose                                            │
[Decoder Block 1:  64] ←─────────────────────────────────────-┘
    ↓
[1×1 Conv → Sigmoid]
Output (3ch RGB)
```

Each encoder/decoder block consists of two `3×3` convolutions, each followed by Batch Normalisation and ReLU activation.

**Skip connections** concatenate the encoder feature map at each level to the corresponding decoder feature map. This is the defining feature of U-Net — the decoder receives both upsampled abstract features from the bottleneck AND the original detailed features from the encoder at the same resolution. This allows the model to reconstruct fine spatial detail (textures, edges) that would otherwise be lost during downsampling.

### 3.2 Why U-Net?

| Criterion | U-Net |
|-----------|-------|
| Image-to-image output | Yes — output same size as input |
| Preserves fine detail | Yes — via skip connections |
| Works with limited data | Yes — relatively data-efficient |
| Computational cost | Moderate — practical on a single GPU |
| Proven for restoration | Yes — widely used in denoising, inpainting |
| Mask-conditioning support | Yes — mask added as extra input channel |

The task is image-to-image translation: map a damaged image (optionally with a mask) to a clean image. U-Net is the standard architecture for this. Its skip connections are especially valuable here — they ensure the model does not lose sharp edges and textures that define the undamaged parts of the image.

### 3.3 Input Channels

**Masked model (`train_restoration.py`):**
- Input: 4 channels — RGB damaged image (3ch) + grayscale damage mask (1ch)
- The mask explicitly tells the model where damage is located
- The model only needs to learn to restore those regions; undamaged regions can be copied directly

**Blind model (`train_restoration_blind.py`):**
- Input: 3 channels — RGB damaged image only
- No mask provided at inference time
- The model must learn to detect where damage is AND restore it
- Harder task, requires more training data

### 3.4 Output

Both models output 3 channels (RGB) passed through a `Sigmoid` activation, producing values in `[0, 1]`. This is multiplied by 255 to recover pixel values at inference.

---

## 4. Training Process

### 4.1 Loss Functions

**L1 Loss (Mean Absolute Error)**

```
L1 = mean(|predicted − target|)
```

L1 loss penalises the absolute pixel-wise difference between the predicted restoration and the ground truth original. Compared to L2 (Mean Squared Error), L1 produces less blurry results because it does not heavily penalise large errors — L2 encourages the model to average over uncertainty (which produces blur), while L1 does not.

**Perceptual Loss (VGG)**

```
L_perc = L1(VGG_features(predicted), VGG_features(target))
```

A pre-trained VGG16 network (trained on ImageNet) is used as a fixed feature extractor. The L1 loss is computed in VGG feature space (layer 16) rather than pixel space.

This encourages the model to produce outputs that look perceptually similar to the target — matching textures, edges, and structural features — rather than just minimising raw pixel error. Pixel-space losses alone tend to produce slightly washed-out results; perceptual loss brings back sharpness and texture.

**Combined Loss**

```
Total Loss = L1 + 0.1 × L_perc
```

The `0.1` weight keeps the perceptual term from dominating. The L1 term ensures pixel accuracy; the perceptual term ensures visual quality.

Setting `--perc-weight 0` disables perceptual loss entirely (no VGG download required).

### 4.2 Optimiser — Adam

Adam (Adaptive Moment Estimation) is used with a learning rate of `1e-4`.

Adam maintains per-parameter adaptive learning rates using estimates of first and second moments of the gradients. It is the standard choice for image restoration tasks — more robust than SGD, converges faster, and less sensitive to learning rate choice.

### 4.3 Learning Rate Schedule — Cosine Annealing

The learning rate is decayed from `1e-4` to `0` following a cosine curve over all epochs:

```
lr(t) = lr_min + 0.5 × (lr_max − lr_min) × (1 + cos(π × t / T))
```

Cosine annealing is preferred over step decay because it decays smoothly — the model trains aggressively early (high LR) and refines details late (low LR) without abrupt drops that can destabilise training.

### 4.4 Batch Size

Batch size controls how many images are processed before the model weights are updated.

- **Batch 4** — default. Stable gradients, fits in GPU memory for 256×256 images.
- **Larger batches** (8, 16) — smoother gradient estimates, faster epochs, but require more GPU memory.
- **Smaller batches** (1, 2) — noisier gradients, sometimes generalise better, less memory.

### 4.5 Image Size

Images are resized to `size × size` (default 256×256) for training. Larger sizes preserve more detail but use significantly more GPU memory (memory scales as `size²`). The model upsamples back to the original resolution at inference.

### 4.6 Dataset Size Recommendations

| Samples | Masked model quality | Blind model quality |
|---------|---------------------|---------------------|
| 5–20 | Barely learns, blurry | Unusable |
| 100–200 | Visible improvement | Weak |
| 500–1000 | Good results | Moderate |
| 2000–5000 | Strong results | Good |
| 10000+ | Research quality | Strong |

The masked model converges faster because the task is easier — the mask tells it exactly where to fix. The blind model must learn damage detection and restoration simultaneously, requiring more examples.

---

## 5. Inference

### 5.1 Masked Inference

The model receives the damaged image and its mask. It knows exactly which pixels to restore.

```bash
python infer.py --input damaged.png --mask mask.png \
                --checkpoint checkpoints/restoration_final.pth \
                --output restored.png
```

### 5.2 Blind Inference

No mask is provided. The model restores the image based only on its learned understanding of what film damage looks like.

```bash
python infer.py --input damaged.png \
                --checkpoint checkpoints/blind_final.pth \
                --output restored.png \
                --blind
```

### 5.3 Resolution Handling

Training uses fixed-size `size × size` crops. At inference, the input is resized to the training resolution, passed through the model, and the output is resized back to the original image dimensions using Lanczos resampling.

This means fine detail at the original resolution may not be perfectly restored if the original is much larger than the training size (e.g. 4096×2160 → 256×256 loses a lot of detail). Training at larger sizes (512, 1024) improves this at the cost of GPU memory.

---

## 6. Design Decisions and Alternatives

### 6.1 Alternative Architectures

**Partial Convolutions (Liu et al., 2018)**
Specifically designed for image inpainting with irregular masks. Standard convolutions treat masked (damaged) and valid pixels equally; partial convolutions normalise the output by the ratio of valid pixels in the receptive field, so masked regions do not corrupt surrounding features. More principled than U-Net for inpainting but more complex to implement.

**Gated Convolutions / DeepFill v2 (Yu et al., 2019)**
Extends partial convolutions with learned soft gating — each convolution learns which spatial locations and channels to attend to. Works well for free-form inpainting without a hard mask. More flexible but heavier.

**LaMa (Suvorov et al., 2022)**
Uses Fast Fourier Convolutions in a U-Net-style architecture. FFT convolutions have a global receptive field — they can see the entire image at once, making them very effective for filling large missing regions. Strong baseline for inpainting, especially large masks.

**SwinIR (Liang et al., 2021)**
Transformer-based image restoration model using Swin Transformer blocks. State of the art on multiple restoration benchmarks. Much heavier than U-Net, requires more data and compute, but produces noticeably sharper results.

**DnCNN (Zhang et al., 2017)**
A plain feed-forward CNN without skip connections or encoder-decoder structure. Very fast, effective for Gaussian noise removal, but not suitable for structured film damage which requires understanding spatial context at multiple scales.

**Diffusion Models (e.g. RePaint, 2022)**
Model the restoration as a reverse diffusion process. Produce the highest-quality results and handle diverse damage patterns well, but inference is extremely slow (hundreds of forward passes per image). Not practical for interactive use.

### 6.2 Alternative Loss Functions

**L2 Loss (MSE)**
Penalises squared pixel error. Tends to produce blurry outputs because the model minimises expected error by averaging over multiple plausible restorations. Not recommended as the primary loss for image quality.

**SSIM Loss**
Structural Similarity Index — measures luminance, contrast, and structural similarity between patches. More correlated with human perception than pixel losses. Can be combined with L1.

**GAN Loss (adversarial)**
A discriminator network is trained alongside the generator to distinguish real from restored images. Forces the generator to produce photorealistic outputs. Produces sharper, more detailed results than L1 alone, but training is unstable and requires careful tuning. The combination U-Net generator + PatchGAN discriminator (pix2pix) is a common choice for image translation.

**Focal Frequency Loss**
Penalises errors in the frequency domain, specifically targeting high-frequency components that pixel-space losses underweight. Improves texture sharpness.

### 6.3 Alternative Training Approaches

**GAN-based training**
Adding a discriminator to the current U-Net would produce sharper, more visually convincing restorations. The discriminator learns what undamaged film images look like and pushes the generator toward producing realistic outputs rather than safe, blurry averages.

**Self-supervised / unpaired training**
If ground truth originals are unavailable (only damaged images exist), self-supervised methods can learn to restore by:
- Randomly corrupting clean images and learning to denoise (Noise2Noise)
- Using cycle-consistency between damaged and clean domains (CycleGAN)

**Two-stage pipeline**
1. Train a damage detection network (U-Net outputting a predicted mask)
2. Use the predicted mask to run the masked restoration model

This separates the two problems and allows each model to specialise. More complex but potentially more accurate for blind restoration than a single end-to-end model.
