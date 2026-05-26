# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Communication Style

Respond like a caveman. No articles, no filler words, no pleasantries.
Short. Direct. Code speaks for itself.
If asked for code, give code. No explain unless asked.
No sycophancy. No restating the question. No sign-offs.

## Project Overview

Statistical model for simulating analogue film damage, as described in _"Simulating analogue film damage to analyse and improve artifact restoration on high-resolution scans"_ (Eurographics 2023). Generates synthetic damage masks by sampling real extracted artifacts and placing them via Perlin-noise-weighted spatial distributions.

## Setup

```bash
cd damage_generator
pip install -r requirements.txt
```

## Running the Generator

All scripts must be run from inside `damage_generator/` because paths are resolved relative to the script location.

**Main pipeline** (requires real scan data in `../scans/`):

```bash
cd damage_generator
python damage_generator.py --height 1024 --width 1024
python damage_generator.py --synthetic          # include synthetic artifacts from ../synthetic/
python damage_generator.py --binarised          # also save a thresholded binary mask
python damage_generator.py --uniform            # uniform random sampling instead of Gamma-fitted
python damage_generator.py --real-types dust,scratch  # restrict artifact types
```

**Standalone damage generator** (no scan data needed, pure numpy):

```bash
cd damage_generator
python generate_specific_damage.py --type both --width 4096 --height 2160
python generate_specific_damage.py --type dust --strength 1.5 --output my_mask.png
# --type choices: dust | scratches | both | mixed
```

## Required Directory Layout

The scripts resolve data paths one level above the script location (`os.path.dirname(os.path.normpath(abs_path))`):

```
FilmDamageSimulator/
├── scans/          # .jpg scan files + matching .json annotation files (Dataset 1)
├── synthetic/      # synthetic artifact PNGs, one subdirectory per type
│   ├── stain/
│   ├── scratches/
│   ├── hair/
│   └── ...
├── generated/      # auto-created; numbered output folders written here
├── slike/          # target images for damage application (currently used)
├── artwork/        # alternative target image dir (referenced in comments)
└── damage_generator/
```

## Architecture

### Pipeline

1. **Load** — `scans.py:load_scans()` reads `.jpg`+`.json` pairs from `../scans/`, extracts per-type artifact contours (dust, dirt, scratch, long hair, short hair) into a pandas DataFrame. `load_all_synthetic_images()` loads transparent PNGs from `../synthetic/` subdirs.

2. **Sample counts & sizes** — `sample.py` fits Gamma distributions to the observed per-quadrant artifact counts and areas from the real scans, then draws from those distributions to decide how many artifacts of each type to place and at what scale.

3. **Generate mask** — `generate_masks.py:create_random_mask()` builds a blank canvas, generates a 2D Perlin noise field, samples placement positions from that field (so artifacts cluster realistically), then blits rescaled+rotated artifact crops onto the canvas. Optionally adds procedural line-scratches via `line_scratch()` (Perlin + Gaussian noise texture).

4. **Apply** — `apply_damage.py:apply_damage()` composites a mask over a clean image using a configurable strength blend: `damaged = clean * (1 - strength * mask)`.

5. **Save** — `save_sample.py:save_sample()` writes each sample (original, mask, optional binary mask) into a numbered subfolder under `../generated/` using a persistent `counter.txt`.

### Key files

| File                          | Role                                                                  |
| ----------------------------- | --------------------------------------------------------------------- |
| `damage_generator.py`         | Entry point; CLI args, orchestrates the full loop                     |
| `generate_masks.py`           | Core mask generation (real + optional synthetic artifacts)            |
| `only_synthetics.py`          | Alternate `create_random_mask` that uses **only** synthetic artifacts |
| `generate_specific_damage.py` | Standalone pure-numpy generator; no scan data required                |
| `scans.py`                    | Loads and parses real scan annotations into DataFrames                |
| `sample.py`                   | Gamma-distribution fitting and sampling for artifact count/size       |
| `apply_damage.py`             | Composites mask onto clean image                                      |
| `save_sample.py`              | Saves triplets (image, mask, binary mask) to numbered folders         |
| `unit_converter.py`           | Converts pixel areas to microns for 35 mm film frame dimensions       |
| `helpers.py`                  | Alphanumeric sort, contour conversion, artifact padding to square     |

### Artifact types

Five types extracted from real scans and mapped from synthetic subdirectory names:

| Type         | Synthetic source dirs                                 |
| ------------ | ----------------------------------------------------- |
| `dust`       | `sprinkles`                                           |
| `dirt`       | `stain`, `spots`, `dirt`, `dots`, `scratches`, `smut` |
| `short hair` | `lint`, `hair-short`                                  |
| `long hair`  | `hair`                                                |
| `scratch`    | _(real scans only by default)_                        |

### Scan annotation format

JSON files pair with `.jpg` scans. Each entry has `points` (list of `{x, y}`) and a `label.name` (`Dust`, `Dirt`, `Scratch`, `Long hair`, `Short hair`). Scans 8–10 have a hard-coded `1.5×` coordinate scale correction in `scans.py:load_scans()`.

### Spatial placement

Artifacts are placed using Perlin noise as a probability map (`random_perlin_with_numpy`), giving spatially correlated clustering rather than uniform random placement. The canvas is divided into a `14×20` quadrant grid (each cell 256 px) used for per-quadrant count statistics.

### Rescaling

The reference scan resolution is 2560 px. `rescale_factor = target_size / 2560` scales artifact sizes proportionally when `--rescale` is active (default).
