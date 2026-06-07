"""
Shared training and evaluation utilities.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.augmentation import GaitAugmentation, generate_occlusion_pair
from src.data.dataset import create_dataset
from src.evaluation.metrics import evaluate_all
from src.losses.center import CenterLoss
from src.losses.consistency import OcclusionConsistencyLoss
from src.losses.triplet import TripletLoss
from src.models import BaselineGaitModel, FullGaitModel


def build_model(config: Dict, variant: str = "baseline") -> torch.nn.Module:
    if variant == "occaware":
        return FullGaitModel(config)
    return BaselineGaitModel(config)


def create_dataloader(config: Dict, split: str = "train", transform=None, shuffle: bool = False) -> DataLoader:
    dataset = create_dataset(config, split=split, transform=transform)
    batch_size_cfg = config.get("training", {}).get("batch_size", [4, 4])
    batch_size = batch_size_cfg[0] * batch_size_cfg[1] if isinstance(batch_size_cfg, list) else int(batch_size_cfg)
    return DataLoader(
        dataset,
        batch_size=min(batch_size, max(1, len(dataset))),
        shuffle=shuffle,
        num_workers=0,
    )


def create_losses(config: Dict, num_classes: int) -> Dict[str, torch.nn.Module]:
    feature_dim = config["model"].get("feature_dim", 256)
    loss_cfg = config.get("loss", {})
    return {
        "triplet": TripletLoss(
            margin=loss_cfg.get("triplet", {}).get("margin", 0.2),
            mining=loss_cfg.get("triplet", {}).get("mining", "hard").replace("-", "_"),
        ),
        "center": CenterLoss(
            num_classes=num_classes,
            feature_dim=feature_dim,
            momentum=loss_cfg.get("center", {}).get("momentum", 0.5),
        ),
        "consistency": OcclusionConsistencyLoss(),
    }


def compute_losses(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    loss_modules: Dict[str, torch.nn.Module],
    config: Dict,
    device: torch.device,
    variant: str = "baseline",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    silhouettes = batch["silhouettes"].to(device)
    labels = batch["label"].to(device)
    embeddings = model(silhouettes)

    triplet_loss, triplet_info = loss_modules["triplet"](embeddings, labels)
    center_loss, center_info = loss_modules["center"](embeddings, labels)

    total_loss = (
        config["loss"]["triplet"].get("weight", 1.0) * triplet_loss
        + config["loss"]["center"].get("weight", 0.005) * center_loss
    )

    metrics = {
        "triplet_loss": float(triplet_loss.item()),
        "center_loss": float(center_loss.item()),
        "num_valid_triplets": float(triplet_info.get("num_valid_triplets", 0)),
        "mean_center_loss": float(center_info.get("center_loss", 0.0)),
    }

    if variant == "occaware" and config["loss"].get("consistency", {}).get("enabled", True):
        frames_np = silhouettes.detach().cpu().numpy()[:, :, 0]
        paired = [generate_occlusion_pair(seq, config.get("augmentation", {}).get("random_occlusion", {})) for seq in frames_np]
        occluded_frames = np.stack([pair[1] for pair in paired], axis=0)
        occluded_tensor = torch.from_numpy(occluded_frames).float().unsqueeze(2).to(device)
        occluded_embeddings = model(occluded_tensor)
        consistency_loss, consistency_info = loss_modules["consistency"](embeddings, occluded_embeddings)
        total_loss = total_loss + config["loss"]["consistency"].get("weight", 0.1) * consistency_loss
        metrics["consistency_loss"] = float(consistency_loss.item())
        metrics["mean_embedding_dist"] = float(consistency_info.get("mean_embedding_dist", 0.0))

    metrics["total_loss"] = float(total_loss.item())
    return total_loss, metrics


def evaluate_model(model: torch.nn.Module, config: Dict, device: torch.device) -> Dict[str, float]:
    gallery_loader = create_dataloader(config, split="gallery")
    probe_loader = create_dataloader(config, split="probe")

    gallery_features, gallery_labels, gallery_views, _ = extract_embeddings(model, gallery_loader, device)
    probe_features, probe_labels, probe_views, probe_categories = extract_embeddings(model, probe_loader, device)

    return evaluate_all(
        query_features=probe_features,
        gallery_features=gallery_features,
        query_labels=probe_labels,
        gallery_labels=gallery_labels,
        query_views=probe_views,
        gallery_views=gallery_views,
        query_categories=probe_categories,
    )


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
):
    model.eval()
    features = []
    labels = []
    views = []
    categories = []

    for batch in dataloader:
        silhouettes = batch["silhouettes"].to(device)
        embeddings = model(silhouettes)
        if embeddings.dim() == 3:
            embeddings = embeddings.mean(dim=1)
        features.append(embeddings.detach().cpu().numpy())
        labels.append(batch["label"].detach().cpu().numpy())
        views.extend(batch["view"])
        categories.extend(batch["category"])

    return (
        np.concatenate(features, axis=0),
        np.concatenate(labels, axis=0),
        np.asarray(views),
        np.asarray(categories),
    )
