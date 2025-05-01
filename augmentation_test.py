# augmentation_test.py

import os
import glob
import random

# Use Agg backend for headless servers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cv2
import albumentations as A
import numpy as np

# 1) define your classes
CLASSES = ['benign', 'malignant', 'normal']
DATASET_DIR = 'Dataset_BUSI_with_GT'

def load_random_samples():
    samples = {}
    for cls in CLASSES:
        # collect all non-mask PNGs in that folder
        folder = os.path.join(DATASET_DIR, cls)
        files = [f for f in glob.glob(os.path.join(folder, '*.png')) if '_mask' not in os.path.basename(f)]
        if not files:
            raise FileNotFoundError(f"No images found for class '{cls}'")
        choice = random.choice(files)
        img = cv2.imread(choice, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to load {choice}")
        samples[cls] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return samples

def make_augmentations():
    return {
        # 1. Orientation invariance
        "HorizontalFlip":       A.HorizontalFlip(p=0.5),
        "VerticalFlip":         A.VerticalFlip(p=0.5),

        # 2. Small rotations (±20°) to mimic probe angle changes
        "Rotate±20°":           A.Rotate(limit=20, p=0.75),

        # 3. Shift & zoom (±10%) for slight repositioning / zoom variability
        "ShiftScaleRotate":     A.ShiftScaleRotate(
                                    shift_limit=0.1,
                                    scale_limit=0.1,
                                    rotate_limit=0,
                                    p=0.75
                                ),

        # 4. Brightness & contrast (±15%) to simulate gain differences
        "Bright/Contrast":      A.RandomBrightnessContrast(
                                    brightness_limit=0.15,
                                    contrast_limit=0.15,
                                    p=0.75
                                ),

        # 5. Speckle noise typical of ultrasound
        "GaussianNoise":        A.GaussNoise(
                                    var_limit=(10.0, 40.0),
                                    mean=0,
                                    p=0.5
                                ),

        # 6. Mild blur (3–7 px) to simulate out-of-focus frames
        "GaussianBlur":         A.GaussianBlur(
                                    blur_limit=(3, 7),
                                    p=0.3
                                ),

        # 7. Local contrast enhancement – CLAHE
        "CLAHE":                A.CLAHE(
                                    clip_limit=3.0,
                                    tile_grid_size=(8,8),
                                    p=0.3
                                ),

        # 8. Elastic warp for small tissue deformations
        "ElasticTransform":     A.ElasticTransform(
                                    alpha=50,
                                    sigma=50,
                                    alpha_affine=15,
                                    p=0.3
                                ),

        # 9. Grid distortion for mild, structured warping
        "GridDistortion":       A.GridDistortion(
                                    num_steps=4,
                                    distort_limit=0.2,
                                    p=0.3
                                ),

        # 10. Small occlusions to mimic signal dropout
        "CoarseDropout":        A.CoarseDropout(
                                    max_holes=4,
                                    max_height=20,
                                    max_width=20,
                                    fill_value=0,
                                    p=0.3
                                ),
    }

def visualize_and_save(img, augs, out_path):
    n = len(augs) + 1
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten()

    # Original
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis('off')

    # Augmented versions
    for ax, (name, aug) in zip(axes[1:], augs.items()):
        out = aug(image=img)['image']
        ax.imshow(out)
        ax.set_title(name)
        ax.axis('off')

    # Hide extras
    for ax in axes[n:]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    print(f" → saved {out_path}")

def main():
    # load one random image per class
    samples = load_random_samples()
    # build your augmentations dict
    augs = make_augmentations()
    # visualize & save per-class
    for cls, img in samples.items():
        out_file = f'augmented_demo_{cls}.png'
        visualize_and_save(img, augs, out_file)

if __name__ == "__main__":
    main()
