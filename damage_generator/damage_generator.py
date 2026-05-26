from scans import load_scans, load_all_synthetic_images
from generate_masks import create_random_mask
import os
import argparse
import cv2 as cv
import uuid
import time
import pandas as pd
from apply_damage import apply_damage
from save_sample import save_sample
start_time = time.time()

import random

def get_random_image(image_dir):
    files = [f for f in os.listdir(image_dir) if f.endswith(".png")]
    return os.path.join(image_dir, random.choice(files))


parser = argparse.ArgumentParser(description='Generate film damage overlays')
parser.add_argument('--height', type=int, nargs='?', const=1024,
                    default=1024, help='height of the mask to be generated')

parser.add_argument('--width', type=int, nargs='?', const=1024,
                    default=1024, help='width of the mask to be generated')

parser.add_argument('--synthetic', action='store_true', help='use additional synthetic damage')
parser.set_defaults(synthetic=False)

parser.add_argument('--rescale', action='store_true', help='rescale artefatcs to match target resolution')
parser.set_defaults(rescale=True)

parser.add_argument('--binarised', action='store_true', help='binarise generated mask')
parser.set_defaults(binarised=False)

parser.add_argument('--verbose', action='store_true')
parser.set_defaults(verbose=True)

parser.add_argument('--uniform', action='store_true', help='uniform sampling instead of using fitted Gamma distributions')
parser.set_defaults(uniform=False)

parser.add_argument('--real-types', type=str, default='all',
                    help='Comma-separated list of real damage types to use. '
                         'Example: dust,scratch,long hair   or "all" for all types')

parser.add_argument('--only-synthetic', action='store_true',
                    help='use only synthetic artifacts — no real scan data required')
parser.set_defaults(only_synthetic=False)

parser.add_argument('--scale', type=float, default=1.0,
                    help='artifact size multiplier (>1 makes artifacts larger, e.g. 2.0 doubles size)')

parser.add_argument('--strength', type=float, default=0.6,
                    help='damage visibility strength 0.0–1.0')

args = parser.parse_args()

if args.only_synthetic:
    args.synthetic = True
    args.uniform = True

abs_path = os.path.abspath(os.path.dirname(__file__))
synthetic_path = os.path.dirname(os.path.normpath(abs_path)) + '/synthetic/'

""" 1. Load real artifacts (skipped in --only-synthetic mode) """
if args.only_synthetic:
    df_artifacts = pd.DataFrame()
    print("Skipping real scan data (--only-synthetic mode)")
else:
    scans_path = os.path.dirname(os.path.normpath(abs_path)) + '/scans/'
    df_artifacts = load_scans(scans_path, verbose=args.verbose)

""" 1.1 Load synthetic artifacts """
if args.synthetic:
    df_synthetic = load_all_synthetic_images(synthetic_path)
else:
    df_synthetic = None


if args.real_types.lower() == 'all':
    enabled_types = ['dust', 'dirt', 'scratch', 'long hair', 'short hair']
else:
    requested = [t.strip().lower() for t in args.real_types.split(',')]
    enabled_types = [t for t in requested if t in ['dust', 'dirt', 'scratch', 'long hair', 'short hair']]
    if not enabled_types:
        print("Warning: No valid real damage types specified. Using none.")

if len(df_artifacts) > 0:
    df_real_filtered = df_artifacts[df_artifacts['Type'].str.lower().isin(enabled_types)].reset_index(drop=True)
else:
    df_real_filtered = pd.DataFrame()

print(f"Enabled real damage types: {enabled_types if enabled_types else 'None'}")
print(f"Total real artifacts after filtering: {len(df_real_filtered)}")

if len(df_real_filtered) > 0:
    df_per_patch_counts = (df_real_filtered.groupby(['Quandrant', 'Type'])
                           .size()
                           .to_frame('Counts')
                           .reset_index())
else:
    df_per_patch_counts = pd.DataFrame(columns=['Quandrant', 'Type', 'Counts'])

directory = "/generated/"
directory = os.path.dirname(os.path.normpath(abs_path)) + directory

if not os.path.exists(directory):
    os.makedirs(directory)

"""2. Generate mask of target size """
"""mask, binary_mask, perlin_noise = create_random_mask(
    (args.height, args.width), 
    df_real_filtered,           # ← Use filtered version instead of full df_artifacts
    df_synthetic,
    df_per_patch_counts,
    use_synthetic=args.synthetic,
    rescale=args.rescale,
    uniform_sample=args.uniform,
    verbose=args.verbose
)
directory = "/generated/"
directory = os.path.dirname(os.path.normpath(abs_path)) + directory

if not os.path.exists(directory):
    os.makedirs(directory)"""

"""counter_file = os.path.join(directory, "counter.txt")

# Load last index
if os.path.exists(counter_file):
    with open(counter_file, "r") as f:
        i = int(f.read().strip())
else:
    i = 0

# Increment
i += 1


with open(counter_file, "w") as f:
    f.write(str(i))

# Use it in filenames
cv.imwrite(f"{directory}damage_mask_{i}.png", mask)

if args.binarised:
    cv.imwrite(f"{directory}binarised_damage_mask_{i}.png", binary_mask)"""

"""# Save mask first (temp path)
temp_mask_path = os.path.join(directory, "temp_mask.png")
cv.imwrite(temp_mask_path, mask)

# Apply mask to image
input_image_path = "../artwork/image0021.jpeg"  # <-- change or randomize later

# Save everything into structured folder
sample_dir = save_sample(
    directory,
    image=cv.imread(input_image_path),  # original image
    mask=mask,
    binary_mask=binary_mask if args.binarised else None
)

# Apply damage and save into same folder
damaged_output_path = os.path.join(sample_dir, "damaged.png")
apply_damage(input_image_path, temp_mask_path, damaged_output_path)

# cleanup temp mask
os.remove(temp_mask_path)
"""

def generate_sample(image_dir, directory):
    # Generate mask
    mask, binary_mask, _ = create_random_mask(
        (args.height, args.width),
        df_real_filtered,
        df_synthetic,
        df_per_patch_counts,
        use_synthetic=args.synthetic,
        rescale=args.rescale,
        uniform_sample=args.uniform,
        verbose=args.verbose,
        enabled_types=enabled_types,
        scale=args.scale,
    )

    # Pick random image
    input_image_path = get_random_image(image_dir)

    # Temp mask
    temp_mask_path = os.path.join(directory, "temp_mask.png")
    cv.imwrite(temp_mask_path, mask)

    # Save base sample (creates folder + counter)
    sample_dir = save_sample(
        directory,
        image=cv.imread(input_image_path),
        mask=mask,
        binary_mask=binary_mask if args.binarised else None
    )

    # Apply damage
    damaged_output_path = os.path.join(sample_dir, "damaged.png")
    apply_damage(input_image_path, temp_mask_path, damaged_output_path, strength=args.strength)

    os.remove(temp_mask_path)
    print(f"Saved sample in {sample_dir}")
    return sample_dir


##image_dir = "../artwork"
image_dir = "../slike"
num_samples = 5  # how many you want

for _ in range(num_samples):
    sample_dir = generate_sample(image_dir, directory)
    print(f"Saved sample in {sample_dir}")

end_time = time.time()
print(f"Total runtime: {end_time - start_time:.2f} seconds")
elapsed = end_time - start_time
print(f"Total runtime: {int(elapsed//60)}m {elapsed%60:.2f}s")