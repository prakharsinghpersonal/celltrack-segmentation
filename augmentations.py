"""
CellTrack — Microscopy-Specific Data Augmentation
Augmentation pipeline designed for microscopy instance segmentation.
Uses albumentations for simultaneous image + mask + bbox transforms.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import config


def get_train_augmentation():
    """Training augmentation pipeline for microscopy images.

    Includes both standard augmentations and microscopy-specific transforms:
    - Geometric: flip, rotate, crop (microscopy images are rotation-invariant)
    - Photometric: brightness, contrast (simulates illumination variation)
    - Noise: Gaussian noise (simulates sensor/detector noise)
    - Elastic: deformation (simulates biological tissue deformation)
    - CLAHE: adaptive contrast (handles uneven illumination)

    Returns:
        albumentations.Compose pipeline with bbox + mask support
    """
    return A.Compose([
        # Geometric transforms — microscopy images have no "up"
        A.RandomCrop(config.AUG_RANDOM_CROP_SIZE, config.AUG_RANDOM_CROP_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        # Photometric transforms — simulates exposure/illumination variation
        A.RandomBrightnessContrast(
            brightness_limit=config.AUG_BRIGHTNESS_LIMIT,
            contrast_limit=config.AUG_CONTRAST_LIMIT,
            p=0.7
        ),

        # Microscopy-specific noise — simulates detector noise
        A.GaussNoise(var_limit=config.AUG_NOISE_VAR_LIMIT, p=0.5),

        # Elastic deformation — simulates biological tissue deformation
        A.ElasticTransform(
            alpha=config.AUG_ELASTIC_ALPHA,
            sigma=config.AUG_ELASTIC_SIGMA,
            p=0.3
        ),

        # Adaptive contrast enhancement
        A.CLAHE(clip_limit=4.0, p=0.3),

        # Normalize and convert to tensor
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='coco',
        label_fields=['labels'],
        min_visibility=0.3
    ))


def get_val_augmentation():
    """Validation/test augmentation — only normalize, no random transforms.

    Returns:
        albumentations.Compose pipeline
    """
    return A.Compose([
        A.Resize(config.AUG_RANDOM_CROP_SIZE, config.AUG_RANDOM_CROP_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='coco',
        label_fields=['labels'],
        min_visibility=0.3
    ))


def apply_augmentation(image, masks, bboxes, labels, transform):
    """Apply augmentation to image, masks, bboxes simultaneously.

    This is why we use albumentations instead of torchvision — it ensures
    geometric transforms are applied consistently to images AND annotations.

    Args:
        image: numpy array (H, W, C)
        masks: list of binary masks, each (H, W)
        bboxes: list of [x, y, w, h] in COCO format
        labels: list of class labels
        transform: albumentations Compose

    Returns:
        Transformed (image, masks, bboxes, labels)
    """
    # Stack masks for simultaneous transformation
    transformed = transform(
        image=image,
        masks=masks,
        bboxes=bboxes,
        labels=labels
    )

    return (
        transformed['image'],
        transformed['masks'],
        transformed['bboxes'],
        transformed['labels']
    )
