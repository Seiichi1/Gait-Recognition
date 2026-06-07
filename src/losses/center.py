"""
Center loss for intra-class compactness.

L_center = (1/2) Σ ||f(x_i) - c_{y_i}||²

Class centers c_j are updated as a moving average of batch features.
Combined with triplet loss for inter-class separation + intra-class compactness.
"""
import torch
import torch.nn as nn
from typing import Tuple


class CenterLoss(nn.Module):
    """Center loss for learning discriminative features.
    
    Maintains learnable class centers and penalizes distance between
    features and their corresponding class centers.
    
    Args:
        num_classes: Number of identity classes.
        feature_dim: Dimensionality of feature embeddings.
        momentum: Moving average momentum for center updates (β). Default: 0.5.
    """
    
    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        momentum: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.momentum = momentum
        
        # Initialize centers
        self.register_buffer(
            'centers',
            torch.randn(num_classes, feature_dim)
        )
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Compute center loss.
        
        Args:
            embeddings: Feature embeddings (B, D) or (B, P, D) for part-based.
            labels: Subject labels (B,).
            
        Returns:
            Tuple of (loss, info_dict).
        """
        if embeddings.dim() == 3:
            # Part-based: (B, P, D) -> compute per part and average
            B, P, D = embeddings.shape
            total_loss = 0
            for p in range(P):
                part_loss, _ = self._compute_center_loss(embeddings[:, p, :], labels)
                total_loss += part_loss
            loss = total_loss / P
        else:
            loss, _ = self._compute_center_loss(embeddings, labels)
        
        info = {'center_loss': loss.item()}
        return loss, info
    
    def _compute_center_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Core center loss computation."""
        # Get centers for each sample's class
        batch_centers = self.centers[labels]  # (B, D)
        
        # Compute L2 distance to centers
        loss = 0.5 * ((embeddings - batch_centers) ** 2).sum(dim=1).mean()
        
        # Update centers with moving average (non-gradient)
        if self.training:
            self._update_centers(embeddings, labels)
        
        return loss, {'center_loss': loss.item()}
    
    @torch.no_grad()
    def _update_centers(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> None:
        """Update class centers using exponential moving average.
        
        c_j^(t+1) = c_j^(t) - β * Δc_j^(t)
        
        where Δc_j = mean(c_j - x_i) for all x_i with label j.
        """
        for j in range(self.num_classes):
            mask = labels == j
            if mask.sum() == 0:
                continue
            
            # Compute delta
            class_features = embeddings[mask]
            delta = self.centers[j] - class_features.mean(dim=0)
            
            # Update center
            self.centers[j] -= self.momentum * delta
