from PIL import Image, ImageOps
import numpy as np
import os

# === CONFIGURATION ===
"""
clean_image_path = "input.jpg"          # ← Change to your photo name
damage_mask_path = "my_damage_mask.png"    # ← Name of the mask you generated
output_path      = "damaged_my_photo.png"

# Load images
clean = Image.open(clean_image_path).convert("RGB")
mask  = Image.open(damage_mask_path).convert("L")   # grayscale

# Resize mask to match your photo size
mask = mask.resize(clean.size, Image.LANCZOS)

# Optional: Make damage more visible by inverting or adjusting intensity
# mask = ImageOps.invert(mask)                    # uncomment if needed
mask = ImageOps.autocontrast(mask)                # increases contrast

# Convert to numpy arrays
clean_np = np.array(clean)
mask_np  = np.array(mask) / 255.0                 # normalize 0-1

# Simple overlay: where mask is bright, make the image darker/damaged
# You can adjust the strength (0.3 = subtle, 0.8 = very damaged)
strength = 0.6
damaged_np = clean_np * (1 - strength * mask_np[..., np.newaxis])

# Convert back to image and save
damaged = Image.fromarray(damaged_np.astype(np.uint8))
damaged.save(output_path)

print(f"✅ Done! Damaged image saved as: {output_path}")
damaged.show()   # opens the result""
"""

def apply_damage(clean_image_path, damage_mask_path, output_path, strength=0.6):
    clean = Image.open(clean_image_path).convert("RGB")
    mask  = Image.open(damage_mask_path).convert("L")

    mask = mask.resize(clean.size, Image.LANCZOS)
    mask = ImageOps.autocontrast(mask)

    clean_np = np.array(clean).astype(float)
    # mask_np: 1 = background, 0 = artifact (inverted mask convention)
    mask_np  = np.array(mask) / 255.0

    factor = mask_np[..., np.newaxis]
    # Artifact areas (factor≈0) are brightened — physical film damage scatters
    # light through scratches/dust, appearing as overexposed bright marks
    damaged_np = np.clip(clean_np + strength * 255 * (1 - factor), 0, 255)

    damaged = Image.fromarray(damaged_np.astype(np.uint8))
    damaged.save(output_path)
    return damaged
