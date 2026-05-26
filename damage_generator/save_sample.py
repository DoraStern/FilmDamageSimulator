import os
import cv2 as cv

def save_sample(base_dir, image, mask, binary_mask=None):
    # persistent counter
    counter_file = os.path.join(base_dir, "counter.txt")

    if os.path.exists(counter_file):
        with open(counter_file, "r") as f:
            i = int(f.read().strip())
    else:
        i = 0

    i += 1

    with open(counter_file, "w") as f:
        f.write(str(i))

    # create folder per sample
    sample_dir = os.path.join(base_dir, f"damage_mask_{i}")
    os.makedirs(sample_dir, exist_ok=True)

    # save files
    cv.imwrite(os.path.join(sample_dir, "image.png"), image)
    cv.imwrite(os.path.join(sample_dir, "mask.png"), mask)

    if binary_mask is not None:
        cv.imwrite(os.path.join(sample_dir, "binary_mask.png"), binary_mask)

    return sample_dir