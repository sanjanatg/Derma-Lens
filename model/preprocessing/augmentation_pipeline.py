"""
augmentation_pipeline.py — Albumentations Augmentation Pipeline
----------------------------------------------------------------
Implements fairness-aware augmentation targeting:
  1. Spatial diversity (rotation, flip, zoom, elastic)
  2. Colour diversity simulating Fitzpatrick I-VI range
  3. Lesion-focused augmentations (cutout, GridMask)
  4. Mixup (implemented at batch level in trainer.py)

Two pipelines:
  - get_train_transforms(): Heavy augmentation for training
  - get_val_transforms():   Minimal (resize + normalise only)
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
INPUT_SIZE    = 224


def get_train_transforms(input_size: int = INPUT_SIZE) -> A.Compose:
    """
    Full augmentation pipeline for training.

    Spatial:
      - Random rotation ±180° (skin lesions are rotation-invariant)
      - Horizontal + vertical flips
      - Random scale (zoom 80–120%)
      - Elastic deformations (simulate tissue deformation)
      - Coarse dropout (random erasing / cutout)

    Colour (tone-balanced):
      - Brightness ±20%
      - Contrast ±20%
      - Hue shift ±10° (within dermatological plausibility)
      - Saturation jitter
      - Random shadow/highlight simulation
      - CLAHE (applied during augmentation too for robustness)

    Returns:
        albumentations.Compose pipeline that returns dict with
        'image' key as uint8 numpy array, converted to normalised tensor.
    """
    return A.Compose([
        # ── Spatial ──────────────────────────────────────────────────
        A.Resize(height=input_size, width=input_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=180, p=0.9),
        A.RandomResizedCrop(
            size=(input_size, input_size),  # Changed: use size instead of height/width
            scale=(0.80, 1.0),
            ratio=(0.9, 1.1),
            p=0.7,
        ),
        A.ElasticTransform(
            alpha=60,
            sigma=6,
            p=0.3,
        ),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),

        # ── Colour / Tone Augmentation ────────────────────────────────
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0,
            ),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,       # ±5% hue — subtle, keeps lesion plausible
                p=1.0,
            ),
        ], p=0.8),

        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.5,
        ),

        # Simulate different lighting conditions (shadows, glare)
        A.RandomShadow(p=0.2),
        A.RandomSunFlare(flare_roi=(0.0, 0.0, 1.0, 0.5), p=0.05),

        # CLAHE within augmentation (mimics preprocessing variation)
        A.CLAHE(clip_limit=2.0, p=0.3),

        # ── Texture / Focus Augmentation ─────────────────────────────
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),    # simulate focus issues
        A.GaussNoise(p=0.2),

        # ── Lesion Occlusion (coarse dropout / cutout) ────────────────
        A.CoarseDropout(p=0.3),

        # ── Normalise and to Tensor ───────────────────────────────────
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(input_size: int = INPUT_SIZE) -> A.Compose:
    """
    Minimal validation/test pipeline.
    Only resize + normalise — no stochastic augmentation.
    """
    return A.Compose([
        A.Resize(height=input_size, width=input_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_tta_transforms(input_size: int = INPUT_SIZE) -> list:
    """
    Test-Time Augmentation (TTA) transforms.
    Returns a list of 8 deterministic pipelines:
      original, h-flip, v-flip, both-flip, +90°, -90°, +90°+hflip, -90°+hflip

    Usage in inference:
        preds = []
        for tta in get_tta_transforms():
            aug = tta(image=img_rgb)['image'].unsqueeze(0)
            preds.append(model(aug))
        final_pred = torch.stack(preds).mean(0)
    """
    base = [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]

    def _compose(geo_transforms):
        return A.Compose(
            [A.Resize(input_size, input_size)] + geo_transforms + base
        )

    return [
        _compose([]),
        _compose([A.HorizontalFlip(p=1.0)]),
        _compose([A.VerticalFlip(p=1.0)]),
        _compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)]),
        _compose([A.Rotate(limit=(90, 90), p=1.0)]),
        _compose([A.Rotate(limit=(-90, -90), p=1.0)]),
        _compose([A.Rotate(limit=(90, 90), p=1.0), A.HorizontalFlip(p=1.0)]),
        _compose([A.Rotate(limit=(-90, -90), p=1.0), A.HorizontalFlip(p=1.0)]),
    ]
