"""
Evaluate a checkpointed gait model.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training import build_model, evaluate_model
from src.utils.checkpoint import load_checkpoint
from src.utils.config import setup_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate gait model")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--variant", choices=["baseline", "occaware"], default="baseline")
    parser.add_argument("--sample_mode", action="store_true", help="Evaluate on synthetic sample data")
    args = parser.parse_args()

    config = setup_experiment(args.config)
    if args.sample_mode:
        config["dataset"]["data_mode"] = "sample"
        config["model"]["pretrained"] = False
        config["model"]["backbone"] = "tiny"
        config["model"]["feature_dim"] = 64

    device = config["device"]
    model = build_model(config, variant=args.variant).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    metrics = evaluate_model(model, config, device)
    print(metrics)


if __name__ == "__main__":
    main()
