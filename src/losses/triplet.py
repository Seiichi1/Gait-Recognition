"""
Triplet loss with hard negative mining for gait recognition.

L_triplet = Σ [||f(x_a) - f(x_p)||² - ||f(x_a) - f(x_n)||² + α]+
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TripletLoss(nn.Module):
    """Triplet loss with online hard/semi-hard negative mining.
    
    Computes all valid triplets within a batch and selects the hardest
    negatives for each anchor-positive pair.
    
    Args:
        margin: Margin α for triplet loss. Default: 0.2.
        mining: Mining strategy - "hard" or "semi_hard". Default: "hard".
    """
    
    def __init__(self, margin: float = 0.2, mining: str = "hard"):
        super().__init__()
        self.margin = margin
        self.mining = mining
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Compute triplet loss with hard negative mining.
        
        Args:
            embeddings: Feature embeddings (B, D) or (B, P, D) for part-based.
            labels: Subject labels (B,).
            
        Returns:
            Tuple of (loss, info_dict) where info_dict contains mining statistics.
        """
        # Handle part-based embeddings
        if embeddings.dim() == 3:
            # (B, P, D) -> compute loss per part and average
            B, P, D = embeddings.shape
            total_loss = 0
            total_info = {'num_valid_triplets': 0, 'num_hard_triplets': 0}
            
            for p in range(P):
                part_loss, part_info = self._compute_triplet_loss(
                    embeddings[:, p, :], labels
                )
                total_loss += part_loss
                total_info['num_valid_triplets'] += part_info['num_valid_triplets']
                total_info['num_hard_triplets'] += part_info['num_hard_triplets']
            
            return total_loss / P, total_info
        else:
            return self._compute_triplet_loss(embeddings, labels)
    
    def _compute_triplet_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Core triplet loss computation."""
        # Compute pairwise distance matrix
        dist_matrix = self._pairwise_distances(embeddings)
        
        B = embeddings.size(0)
        
        # Create masks for valid positive and negative pairs
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        labels_not_equal = ~labels_equal
        
        # Mask out diagonal (anchor != positive requirement)
        not_self = ~torch.eye(B, dtype=torch.bool, device=embeddings.device)
        positive_mask = labels_equal & not_self
        negative_mask = labels_not_equal
        
        if self.mining == "hard":
            loss, info = self._hard_mining(dist_matrix, positive_mask, negative_mask)
        elif self.mining == "semi_hard":
            loss, info = self._semi_hard_mining(dist_matrix, positive_mask, negative_mask)
        else:
            raise ValueError(f"Unknown mining strategy: {self.mining}")
        
        return loss, info
    
    def _hard_mining(
        self,
        dist_matrix: torch.Tensor,
        positive_mask: torch.Tensor,
        negative_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Batch-all hard negative mining.
        
        For each anchor:
        - Select hardest positive (max distance among positives)
        - Select hardest negative (min distance among negatives)
        """
        # Hardest positive for each anchor
        pos_dists = dist_matrix * positive_mask.float()
        hardest_pos, _ = pos_dists.max(dim=1)  # (B,)
        
        # Hardest negative for each anchor
        # Set non-negative pairs to large value before taking min
        neg_dists = dist_matrix + (~negative_mask).float() * 1e6
        hardest_neg, _ = neg_dists.min(dim=1)  # (B,)
        
        # Compute triplet loss
        losses = F.relu(hardest_pos - hardest_neg + self.margin)
        
        # Only count valid anchors (those with at least one positive)
        has_positive = positive_mask.any(dim=1)
        has_negative = negative_mask.any(dim=1)
        valid = has_positive & has_negative
        
        num_valid = valid.sum().item()
        num_hard = (losses[valid] > 0).sum().item() if num_valid > 0 else 0
        
        loss = losses[valid].mean() if num_valid > 0 else torch.tensor(0.0, device=dist_matrix.device)
        
        info = {
            'num_valid_triplets': num_valid,
            'num_hard_triplets': num_hard,
            'mean_pos_dist': hardest_pos[valid].mean().item() if num_valid > 0 else 0,
            'mean_neg_dist': hardest_neg[valid].mean().item() if num_valid > 0 else 0,
        }
        
        return loss, info
    
    def _semi_hard_mining(
        self,
        dist_matrix: torch.Tensor,
        positive_mask: torch.Tensor,
        negative_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Semi-hard negative mining.
        
        Select negatives that are farther than the positive but within the margin:
        d(a, p) < d(a, n) < d(a, p) + margin
        """
        B = dist_matrix.size(0)
        losses = []
        
        for i in range(B):
            pos_indices = positive_mask[i].nonzero(as_tuple=True)[0]
            neg_indices = negative_mask[i].nonzero(as_tuple=True)[0]
            
            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue
            
            for p_idx in pos_indices:
                d_ap = dist_matrix[i, p_idx]
                neg_dists = dist_matrix[i, neg_indices]
                
                # Semi-hard: d_ap < d_an < d_ap + margin
                semi_hard_mask = (neg_dists > d_ap) & (neg_dists < d_ap + self.margin)
                
                if semi_hard_mask.any():
                    d_an = neg_dists[semi_hard_mask].min()
                else:
                    # Fallback to hardest negative
                    d_an = neg_dists.min()
                
                triplet_loss = F.relu(d_ap - d_an + self.margin)
                losses.append(triplet_loss)
        
        if losses:
            loss = torch.stack(losses).mean()
            num_hard = sum(1 for l in losses if l.item() > 0)
        else:
            loss = torch.tensor(0.0, device=dist_matrix.device)
            num_hard = 0
        
        return loss, {'num_valid_triplets': len(losses), 'num_hard_triplets': num_hard}
    
    @staticmethod
    def _pairwise_distances(embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise Euclidean distance matrix.
        
        Args:
            embeddings: (B, D) tensor.
            
        Returns:
            (B, B) distance matrix.
        """
        dot_product = torch.mm(embeddings, embeddings.t())
        sq_norms = dot_product.diagonal()
        distances = sq_norms.unsqueeze(0) - 2.0 * dot_product + sq_norms.unsqueeze(1)
        distances = F.relu(distances)  # Numerical stability
        distances = torch.sqrt(distances + 1e-8)
        return distances
