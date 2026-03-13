"""
CellTrack — Utility Functions
Visualization, logging, and helper functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import torch


def visualize_prediction(image, prediction, score_threshold=0.5, save_path=None):
    """Visualize Mask R-CNN predictions on an image.

    Args:
        image: numpy array (H, W, 3) in RGB
        prediction: dict with 'boxes', 'labels', 'scores', 'masks'
        score_threshold: Minimum confidence to display
        save_path: Optional path to save the figure
    """
    boxes = prediction['boxes'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    masks = prediction['masks'].cpu().numpy()

    # Filter by score
    keep = scores >= score_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    masks = masks[keep]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Original image with boxes
    axes[0].imshow(image)
    axes[0].set_title('Detections')
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(boxes), 1)))

    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=colors[i], facecolor='none'
        )
        axes[0].add_patch(rect)
        axes[0].text(x1, y1 - 5, f'{score:.2f}',
                     color=colors[i], fontsize=10, fontweight='bold')

    # Mask overlay
    axes[1].imshow(image)
    axes[1].set_title('Instance Masks')
    combined_mask = np.zeros((*image.shape[:2], 4))

    for i, mask in enumerate(masks):
        binary_mask = mask[0] > 0.5  # Threshold mask
        color = list(colors[i][:3]) + [0.4]  # RGBA with transparency
        mask_overlay = np.zeros((*image.shape[:2], 4))
        mask_overlay[binary_mask] = color
        combined_mask = np.maximum(combined_mask, mask_overlay)

    axes[1].imshow(combined_mask)

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_augmentation(original, augmented, title="Augmentation Example",
                           save_path=None):
    """Side-by-side visualization of original and augmented images.

    Args:
        original: Original image (H, W) or (H, W, 3)
        augmented: Augmented image
        title: Plot title
        save_path: Optional save path
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(original, cmap='gray' if len(original.shape) == 2 else None)
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(augmented, cmap='gray' if len(augmented.shape) == 2 else None)
    axes[1].set_title('Augmented')
    axes[1].axis('off')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(train_losses, val_maps, save_path=None):
    """Plot training loss and validation mAP over epochs.

    Divergence between training and validation = overfitting signal.

    Args:
        train_losses: List of training losses per epoch
        val_maps: Dict of epoch -> mAP
        save_path: Optional save path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Training loss
    ax1.plot(train_losses, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)

    # Validation mAP
    if val_maps:
        epochs = sorted(val_maps.keys())
        maps = [val_maps[e] for e in epochs]
        ax2.plot(epochs, maps, 'g-o', linewidth=2, markersize=6)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mAP')
        ax2.set_title('Validation mAP')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.82, color='r', linestyle='--', alpha=0.5, label='Target: 0.82')
        ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
