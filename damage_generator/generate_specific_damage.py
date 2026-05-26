import numpy as np
from PIL import Image
import argparse
import os

# ====================== CONFIG ======================
WIDTH = 4096
HEIGHT = 2160
OUTPUT_DIR = "."

# Density / intensity for each type (higher = more damage)
DUST_DENSITY     = 0.0008      # small specks and stains
SCRATCH_DENSITY  = 0.00015     # long thin scratches / hairs
# ===================================================

def generate_dust_mask(height, width, density=0.0008, max_size=8):
    mask = np.zeros((height, width), dtype=np.float32)
    num_particles = int(height * width * density)
    
    y = np.random.randint(0, height, num_particles)
    x = np.random.randint(0, width, num_particles)
    sizes = np.random.randint(1, max_size+1, num_particles)
    
    for i in range(num_particles):
        r = sizes[i]
        yy, xx = np.ogrid[-r:r+1, -r:r+1]
        mask_region = mask[max(0, y[i]-r):min(height, y[i]+r+1),
                           max(0, x[i]-r):min(width, x[i]+r+1)]
        circle = (yy**2 + xx**2) <= r**2
        mask_region[:circle.shape[0], :circle.shape[1]] = np.maximum(mask_region[:circle.shape[0], :circle.shape[1]], circle.astype(float))
    
    return mask


def generate_scratch_mask(height, width, density=0.00015, min_length=200, max_length=1800):
    mask = np.zeros((height, width), dtype=np.float32)
    num_scratches = int(height * width * density * 300)   # empirical multiplier for visibility
    
    for _ in range(num_scratches):
        x1 = np.random.randint(0, width)
        y1 = np.random.randint(0, height)
        length = np.random.randint(min_length, max_length)
        angle = np.random.uniform(0, 2 * np.pi)
        
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        
        # Draw thick line
        thickness = np.random.randint(1, 4)
        steps = max(abs(x2 - x1), abs(y2 - y1)) + 1
        
        for t in range(steps):
            xt = int(x1 + t * (x2 - x1) / steps)
            yt = int(y1 + t * (y2 - y1) / steps)
            if 0 <= xt < width and 0 <= yt < height:
                for dy in range(-thickness, thickness+1):
                    for dx in range(-thickness//2, thickness//2 + 1):
                        if 0 <= yt+dy < height and 0 <= xt+dx < width:
                            mask[yt+dy, xt+dx] = 1.0
    return mask


def main():
    parser = argparse.ArgumentParser(description="Generate specific film damage types")
    parser.add_argument("--type", choices=["dust", "scratches", "both", "mixed"], default="both",
                        help="Type of damage to generate: dust, scratches, both, or mixed")
    parser.add_argument("--width", type=int, default=4096, help="Width of mask")
    parser.add_argument("--height", type=int, default=2160, help="Height of mask")
    parser.add_argument("--output", type=str, default=None, help="Output filename")
    parser.add_argument("--strength", type=float, default=1.0, help="Overall strength multiplier")
    
    args = parser.parse_args()

    width = args.width
    height = args.height
    
    print(f"Generating {args.type} damage mask ({width}x{height})...")

    dust_mask = np.zeros((height, width), dtype=np.float32)
    scratch_mask = np.zeros((height, width), dtype=np.float32)

    if args.type in ["dust", "both", "mixed"]:
        dust_mask = generate_dust_mask(height, width, density=DUST_DENSITY)
    
    if args.type in ["scratches", "both", "mixed"]:
        scratch_mask = generate_scratch_mask(height, width, density=SCRATCH_DENSITY)

    # Combine
    if args.type == "mixed":
        final_mask = dust_mask * 0.7 + scratch_mask * 1.0   # mixed with different weights
    else:
        final_mask = dust_mask + scratch_mask

    final_mask = np.clip(final_mask * args.strength, 0, 1)

    # Save as 8-bit image
    final_img = (final_mask * 255).astype(np.uint8)
    mask_pil = Image.fromarray(final_img)

    if args.output is None:
        filename = f"damage_{args.type}_{width}x{height}.png"
    else:
        filename = args.output

    output_path = os.path.join(OUTPUT_DIR, filename)
    mask_pil.save(output_path)
    
    print(f"✅ {args.type.upper()} mask saved as: {output_path}")
    mask_pil.show()   # opens preview


if __name__ == "__main__":
    main()