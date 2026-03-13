"""
CellTrack — Training Loop
End-to-end training with LR scheduling, checkpointing, and logging.
"""

import os
import time
import torch
from torch.utils.data import DataLoader

import config
from model import get_model, freeze_backbone, get_trainable_params
from dataset import MicroscopyDataset, collate_fn
from evaluate import evaluate_model


def train_one_epoch(model, loader, optimizer, device, epoch):
    """Train for one epoch.

    Mask R-CNN returns a dict of losses during training:
    - loss_classifier: classification loss
    - loss_box_reg: bounding box regression loss
    - loss_mask: mask prediction loss
    - loss_objectness: RPN objectness loss
    - loss_rpn_box_reg: RPN box regression loss

    Args:
        model: Mask R-CNN model
        loader: Training DataLoader
        optimizer: Optimizer
        device: torch device
        epoch: Current epoch number

    Returns:
        Average total loss for the epoch
    """
    model.train()
    total_loss = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass — Mask R-CNN returns losses in training mode
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

        if batch_idx % 10 == 0:
            loss_str = ", ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
            print(f"  Epoch {epoch} [{batch_idx}/{len(loader)}] — {loss_str}")

    return total_loss / len(loader)


def train(data_dir=None, num_epochs=None, batch_size=None, resume_from=None):
    """Full training pipeline.

    Args:
        data_dir: Dataset root directory
        num_epochs: Number of training epochs
        batch_size: Batch size
        resume_from: Path to checkpoint to resume from
    """
    data_dir = data_dir or config.DATA_DIR
    num_epochs = num_epochs or config.NUM_EPOCHS
    batch_size = batch_size or config.BATCH_SIZE
    device = config.DEVICE

    print(f"Training on: {device}")
    print(f"Epochs: {num_epochs}, Batch size: {batch_size}")

    # Dataset & DataLoader
    train_dataset = MicroscopyDataset(os.path.join(data_dir, "train"), split="train")
    val_dataset = MicroscopyDataset(os.path.join(data_dir, "val"), split="val")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn,
        pin_memory=config.PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn
    )

    # Model
    model = get_model(num_classes=config.NUM_CLASSES, pretrained=config.PRETRAINED)
    freeze_backbone(model, freeze_layers=2)
    model.to(device)

    # Optimizer — SGD with momentum + weight decay
    params = get_trainable_params(model)
    optimizer = torch.optim.SGD(
        params, lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY
    )

    # Learning rate scheduler — cosine annealing for smooth convergence
    if config.LR_SCHEDULER == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA
        )

    # Resume from checkpoint
    start_epoch = 0
    best_map = 0.0

    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_map = checkpoint.get("best_map", 0.0)
        print(f"Resumed from epoch {start_epoch}, best mAP: {best_map:.4f}")

    # Create checkpoint directory
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        t_start = time.time()

        # Train
        avg_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        scheduler.step()

        t_end = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{num_epochs} — Loss: {avg_loss:.4f}, "
              f"LR: {current_lr:.6f}, Time: {t_end - t_start:.1f}s")

        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            metrics = evaluate_model(model, val_loader, device)
            current_map = metrics.get("mAP", 0.0)
            print(f"  Validation mAP: {current_map:.4f}")

            # Save best model
            if current_map > best_map:
                best_map = current_map
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_map": best_map,
                    "config": {
                        "num_classes": config.NUM_CLASSES,
                        "backbone": config.BACKBONE,
                    }
                }, os.path.join(config.CHECKPOINT_DIR, "best_model.pth"))
                print(f"  ✓ New best model saved (mAP: {best_map:.4f})")

        # Save periodic checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_map": best_map,
        }, os.path.join(config.CHECKPOINT_DIR, "latest_checkpoint.pth"))

    print(f"\nTraining complete. Best mAP: {best_map:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train CellTrack Mask R-CNN")
    parser.add_argument("--data_dir", default=config.DATA_DIR)
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    train(args.data_dir, args.epochs, args.batch_size, args.resume)
