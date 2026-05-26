# Simulating analogue film damage to analyse and improve artefact restoration on high-resolution scans

[[Paper]](https://arxiv.org/pdf/2302.10004.pdf) [[Project Page]](https://daniela997.github.io/FilmDamageSimulator/) [[Dataset 1]](https://doi.org/10.6084/m9.figshare.21803304.v2) [[Dataset 2]](https://doi.org/10.6084/m9.figshare.21803292)


![overview](https://user-images.githubusercontent.com/32989037/223543778-a548271f-0cda-493f-91cf-2c38aa5c36cc.png)

## Film Damage Simulator

Statistical model for simulating analogue film damage. Generates synthetic damage masks and applies them to clean images to produce training data for film artifact restoration.

---

## Setup

### Requirements

- Python 3.10+
- NVIDIA GPU recommended for training

### 1. Clone

```bash
git clone https://github.com/YOUR_USERNAME/FilmDamageSimulator.git
cd FilmDamageSimulator
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install PyTorch

Pick the command matching your setup from [pytorch.org](https://pytorch.org/get-started/locally/).

**CUDA 12.4 (NVIDIA GPU):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**CPU only:**
```bash
pip install torch torchvision
```

---

## Generating Damage Masks

All generator scripts run from inside `damage_generator/`.

```bash
cd damage_generator
```

**No scan data needed (synthetic only):**
```bash
python damage_generator.py --only-synthetic --scale 2.0 --height 512 --width 512
```

**With real scan data in `../scans/`:**
```bash
python damage_generator.py --height 1024 --width 1024
python damage_generator.py --synthetic          # add synthetic artifacts
python damage_generator.py --real-types dust,scratch  # specific types only
python damage_generator.py --binarised          # also save binary mask
```

**Standalone numpy generator (no dependencies beyond numpy):**
```bash
python generate_specific_damage.py --type both --width 1920 --height 1080
python generate_specific_damage.py --type dust --strength 1.5 --output my_mask.png
# --type: dust | scratches | both | mixed
```

Output saved to `../generated/damage_mask_N/` with `image.png`, `mask.png`, `damaged.png`.

---

## Training the Restoration Model

From `damage_generator/`:

```bash
# Generate training data first (run multiple times for more samples)
python damage_generator.py --only-synthetic --scale 2.0

# Train
python train_restoration.py --data-dir ../generated --epochs 50 --batch-size 4 --perc-weight 0
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `generated` | path to generated samples |
| `--epochs` | 50 | training epochs |
| `--batch-size` | 4 | images per batch (reduce if out of GPU memory) |
| `--size` | 256 | resize images to N×N for training |
| `--perc-weight` | 0.1 | perceptual loss weight (set 0 to skip VGG download) |
| `--lr` | 1e-4 | learning rate |
| `--save-every` | 10 | save checkpoint every N epochs |

Checkpoints saved to `checkpoints/`. Final model: `checkpoints/restoration_final.pth`.

### Windows GPU note

If training freezes your PC, increase the GPU watchdog timeout (run PowerShell as Administrator, then reboot):

```powershell
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" -Name TdrDelay -Value 60
Restart-Computer
```

---

## Directory Layout

```
FilmDamageSimulator/
├── damage_generator/     # damage generation + training scripts
├── scans/                # (optional) real scan .jpg + .json annotation pairs
├── synthetic/            # (optional) synthetic artifact PNGs by type
├── generated/            # output — created automatically
└── slike/                # target images for damage application
```
