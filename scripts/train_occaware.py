"""
Train the occlusion-aware gait model.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training import build_model, compute_losses, create_dataloader, create_losses, evaluate_model
from src.utils.checkpoint import save_checkpoint
from src.utils.config import setup_experiment
from src.utils.logging_utils import finish_logging, log_metrics, setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Train occlusion-aware gait model")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--smoke_test", action="store_true", help="Run a short synthetic verification loop")
    args = parser.parse_args()

    config = setup_experiment(args.config)
    if args.smoke_test:
        config["dataset"]["data_mode"] = "sample"
        config["model"]["pretrained"] = False
        config["model"]["backbone"] = "tiny"
        config["model"]["feature_dim"] = 64
        config["training"]["batch_size"] = [2, 2]

    device = config["device"]
    model = build_model(config, variant="occaware").to(device)
    logger = setup_logger(config)
    train_loader = create_dataloader(
        config,
        split="train",
        transform=GaitAugmentation(config.get("augmentation", {})),
        shuffle=True,
    )
    num_classes = getattr(train_loader.dataset, "num_classes", 4)
    losses = create_losses(config, num_classes=num_classes)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["training"]["optimizer"].get("lr", 0.01),
        momentum=config["training"]["optimizer"].get("momentum", 0.9),
        weight_decay=config["training"]["optimizer"].get("weight_decay", 5e-4),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, config["training"]["scheduler"].get("milestones", [10])[0]),
        gamma=config["training"]["scheduler"].get("gamma", 0.1),
    )

    iterations = 3 if args.smoke_test else config["training"].get("total_iterations", 100)
    model.train()

    for step, batch in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        total_loss, metrics = compute_losses(model, batch, losses, config, device, variant="occaware")
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"].get("max_norm", 10.0))
        optimizer.step()
        scheduler.step()
        log_metrics(logger, metrics, step, backend=config["logging"].get("backend", "wandb"))
        if step >= iterations:
            break

    eval_metrics = evaluate_model(model, config, device)
    save_checkpoint(model, optimizer, scheduler, iterations, eval_metrics, config, save_dir="checkpoints", filename="occaware_smoke.pth")
    finish_logging(logger, backend=config["logging"].get("backend", "wandb"))
    print("Occlusion-aware training complete")
    print(eval_metrics)


if __name__ == "__main__":
    from src.data.augmentation import GaitAugmentation

    main()
