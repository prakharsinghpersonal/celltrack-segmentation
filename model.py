"""
CellTrack — Mask R-CNN Model Setup
Fine-tuning pretrained Mask R-CNN on microscopy data.
"""

import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import config


def get_model(num_classes=None, pretrained=True):
    """Build Mask R-CNN with pretrained ResNet-50 FPN backbone.

    Architecture:
    - ResNet-50 backbone (pretrained on ImageNet) — extracts hierarchical features
    - FPN (Feature Pyramid Network) — multi-scale feature maps for detecting
      objects of different sizes
    - RPN (Region Proposal Network) — generates candidate bounding boxes
    - RoI Align — extracts fixed-size features per proposal (bilinear interpolation)
    - Classification head — predicts class per proposal
    - Bounding box regression head — refines box coordinates
    - Mask head — predicts per-pixel binary mask per proposal

    Why Mask R-CNN for microscopy:
    1. Instance segmentation — distinguishes overlapping cells (unlike U-Net)
    2. Pretrained backbone — ImageNet features transfer to microscopy surprisingly well
    3. FPN — handles cells at multiple scales in the same image

    Args:
        num_classes: Number of classes (including background)
        pretrained: Use ImageNet pretrained weights

    Returns:
        Mask R-CNN model ready for fine-tuning
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES

    # Load pretrained Mask R-CNN
    if pretrained:
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        model = maskrcnn_resnet50_fpn(weights=weights)
    else:
        model = maskrcnn_resnet50_fpn(weights=None)

    # Replace the box classifier head for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace the mask predictor head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


def freeze_backbone(model, freeze_layers=4):
    """Freeze early backbone layers for transfer learning.

    Early layers learn universal features (edges, textures) that transfer
    well from ImageNet to microscopy. Freezing them:
    - Prevents catastrophic forgetting
    - Reduces training time
    - Reduces overfitting on small datasets

    Args:
        model: Mask R-CNN model
        freeze_layers: Number of ResNet layers to freeze (0-4)
    """
    # Freeze batch norm everywhere
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.requires_grad_(False)

    # Freeze ResNet layers
    backbone = model.backbone.body
    layers_to_freeze = [backbone.conv1, backbone.bn1]
    layer_groups = [backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4]

    for i, layer in enumerate(layers_to_freeze):
        for param in layer.parameters():
            param.requires_grad = False

    for i in range(min(freeze_layers, len(layer_groups))):
        for param in layer_groups[i].parameters():
            param.requires_grad = False


def get_trainable_params(model):
    """Get only the parameters that require gradients.

    Returns:
        List of trainable parameters for optimizer
    """
    return [p for p in model.parameters() if p.requires_grad]


def count_parameters(model):
    """Count total and trainable parameters.

    Returns:
        (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    model = get_model(num_classes=2, pretrained=True)
    freeze_backbone(model, freeze_layers=2)

    total, trainable = count_parameters(model)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters:    {total - trainable:,}")
