"""
CellTrack — Custom PyTorch Dataset for Microscopy
Loads microscopy images + annotations with augmentation support.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2

from augmentations import get_train_augmentation, get_val_augmentation
from preprocess import preprocess_pipeline


class MicroscopyDataset(Dataset):
    """PyTorch Dataset for microscopy instance segmentation.

    Expects data organized as:
        data_dir/
        ├── images/
        │   ├── img_001.png
        │   └── ...
        └── annotations/
            └── instances.json  (COCO format)

    Args:
        data_dir: Root directory containing images/ and annotations/
        split: 'train' or 'val'
        preprocess: Whether to apply preprocessing (CLAHE, denoising)
    """

    def __init__(self, data_dir, split="train", preprocess=True):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "images")
        self.split = split
        self.preprocess = preprocess

        # Load COCO-format annotations
        ann_file = os.path.join(data_dir, "annotations", "instances.json")
        if os.path.exists(ann_file):
            with open(ann_file, "r") as f:
                self.coco_data = json.load(f)
            self.images = self.coco_data.get("images", [])
            self.annotations = self._build_annotation_index()
        else:
            # Fallback: load images without annotations (inference mode)
            self.images = self._discover_images()
            self.annotations = {}

        # Augmentation pipeline
        if split == "train":
            self.transform = get_train_augmentation()
        else:
            self.transform = get_val_augmentation()

    def _discover_images(self):
        """Discover images in directory when no annotation file exists."""
        extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        images = []
        for f in sorted(os.listdir(self.image_dir)):
            if os.path.splitext(f)[1].lower() in extensions:
                images.append({"id": len(images), "file_name": f})
        return images

    def _build_annotation_index(self):
        """Build image_id -> annotations lookup for efficient access."""
        index = {}
        for ann in self.coco_data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in index:
                index[img_id] = []
            index[img_id].append(ann)
        return index

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """Load image and annotations, apply transforms.

        Returns:
            image: Tensor (C, H, W)
            target: dict with keys: boxes, labels, masks, image_id, area, iscrowd
        """
        img_info = self.images[idx]
        img_path = os.path.join(self.image_dir, img_info["file_name"])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess
        if self.preprocess:
            image = preprocess_pipeline(image)
            image = (image * 255).astype(np.uint8)

        # Load annotations
        img_id = img_info["id"]
        anns = self.annotations.get(img_id, [])

        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []

        for ann in anns:
            # COCO bbox: [x, y, width, height] -> [x1, y1, x2, y2]
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann.get("category_id", 1))
            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))

            # Decode RLE mask or polygon to binary mask
            if "segmentation" in ann:
                mask = self._decode_mask(ann["segmentation"], image.shape[:2])
                masks.append(mask)

        # Convert to tensors
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, *image.shape[:2]), dtype=torch.uint8)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": iscrowd,
        }

        # Convert image to tensor
        image = torch.as_tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0

        return image, target

    def _decode_mask(self, segmentation, image_size):
        """Decode COCO segmentation (polygon or RLE) to binary mask."""
        mask = np.zeros(image_size, dtype=np.uint8)
        if isinstance(segmentation, list):
            # Polygon format
            for polygon in segmentation:
                pts = np.array(polygon).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)
        return mask


def collate_fn(batch):
    """Custom collate function for detection datasets.
    
    Each sample has variable number of objects, so we can't stack targets.
    """
    return tuple(zip(*batch))
