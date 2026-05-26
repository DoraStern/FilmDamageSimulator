import numpy as np
import random
import pandas as pd
import math
import skimage.transform as skimage_tf
import cv2 as cv
from sample import sample_num_artifacts, sample_size_artifacts, sample_closest_in_area

def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)


def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency*res[0], frequency*res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise

def increase_contrast(pixvals):
    minval = np.percentile(pixvals, 2)
    maxval = np.percentile(pixvals, 98)
    pixvals = np.clip(pixvals, minval, maxval)
    pixvals = ((pixvals - minval) / (maxval - minval))
    return pixvals

def random_perlin_with_numpy(num_samples, noise_array):
    # Create a flat copy of the array
    linear_idx = np.random.choice(noise_array.size, p=noise_array.ravel()/float(noise_array.sum()), size=num_samples)
    x, y = np.unravel_index(linear_idx, noise_array.shape)
    return x, y

def shift_bit_length(x):
    return 1<<(x-1).bit_length()

def line_scratch(length):
    length_pot = shift_bit_length(length.item())
    noise_scale = np.random.randint(1, 4, size=1, dtype=int)[0].item()
    perlin_noise = generate_perlin_noise_2d((length_pot, length_pot), (2**noise_scale, 2**noise_scale))
    normalised_noise = (perlin_noise - np.min(perlin_noise))/np.ptp(perlin_noise)
    slice_fade = np.random.randint(low=0, high=length-20, size=1, dtype=int)[0]
    fade = increase_contrast(normalised_noise)[0:length.item(), slice_fade.item():slice_fade.item()+20]
    fade *= (255.0/fade.max())
    mean = 0
    var = 15
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, (fade.shape[0], fade.shape[1])) 
    fade = fade + gaussian
    fade = fade.astype(np.uint8)

    num_lines = np.random.randint(1, 2, size=1, dtype=int)
    lines_xs = np.random.randint(1, 19, size=num_lines, dtype=int)
    lines_ys = np.random.randint(0, int(length*0.2), size=num_lines, dtype=int)

    line_mask_width, line_mask_height = 20, length
    line_mask = np.ones((line_mask_height, line_mask_width)) * 255
    line_thickness = 1

    
    x1, y1 = lines_xs[0], lines_ys[0]
    x2, y2 = lines_xs[0], np.random.randint(lines_ys[0]+10, length-(lines_ys[0]+10), size=1, dtype=int)
    line = cv.line(line_mask, (x1, y1), (x2, int(y2)), (0, 255, 0), thickness=line_thickness).astype(np.uint8)

    scratch = cv.bitwise_and(fade, np.invert(line)).astype(np.uint8)
    scratch = cv.GaussianBlur(scratch, (3,3), 0).astype(np.uint8)
    scratch = cv.resize(scratch, None, fx = 0.5, fy = 1, interpolation = cv.INTER_CUBIC)

    return scratch
def create_random_mask(
    target_size,
    artifact_library,
    min_artifacts=20,
    max_artifacts=120,
    rescale=True,
    verbose=False
):
    if rescale:
        rescale_factor = min(target_size) / 2560
    else:
        rescale_factor = 1.0

    artifacts_num = np.random.randint(min_artifacts, max_artifacts)

    types = list(artifact_library.keys())

    selected_artifacts = [
        random.choice(artifact_library[random.choice(types)])
        for _ in range(artifacts_num)
    ]

    mask_final = np.zeros(target_size, dtype=np.uint8)

    perlin_noise = generate_perlin_noise_2d(target_size, (2, 2))
    normalised_noise = (perlin_noise - np.min(perlin_noise)) / np.ptp(perlin_noise)
    xs, ys = random_perlin_with_numpy(artifacts_num, normalised_noise)

    random_angles = np.random.randint(0, 360, size=artifacts_num)

    for i, artifact in enumerate(selected_artifacts):
        try:
            scale = np.random.uniform(0.3, 1.5) * rescale_factor
            angle = random_angles[i]

            artifact = skimage_tf.rescale(
                artifact.astype(np.uint8),
                scale,
                preserve_range=True,
                anti_aliasing=True
            )

            artifact = skimage_tf.rotate(
                artifact,
                angle=angle,
                resize=True,
                preserve_range=True
            ).astype(np.uint8)

            h, w = artifact.shape[:2]

            x1 = xs[i] - h // 2
            y1 = ys[i] - w // 2
            x2 = x1 + h
            y2 = y1 + w

            if x1 < 0:
                artifact = artifact[-x1:, :]
                x1 = 0
            if y1 < 0:
                artifact = artifact[:, -y1:]
                y1 = 0
            if x2 > target_size[0]:
                artifact = artifact[:target_size[0] - x1, :]
                x2 = target_size[0]
            if y2 > target_size[1]:
                artifact = artifact[:, :target_size[1] - y1]
                y2 = target_size[1]

            mask_final[x1:x2, y1:y2] = np.maximum(
                mask_final[x1:x2, y1:y2],
                artifact
            )

        except Exception:
            continue

    mask_final = np.invert(mask_final)

    thresh = 240
    binarised_mask_final = (mask_final > thresh) * 255

    return mask_final.astype(np.uint8), binarised_mask_final.astype(np.uint8)