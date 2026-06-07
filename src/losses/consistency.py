"""
Occlusion-aware consistency regularization.

L_consistency = Σ ||f(x_clean) - f(x_occluded)||²

Novel regularization that enforces embedding similarity across different
occlusion patterns of the same subject, trained via synthetic paired samples.
"""
import torch
import torch.nn as nn
from typing import Tuple


class OcclusionConsistencyLoss(nn.Module):
    """Consistency loss for occlusion invariance.
    
    Enforces that embeddings of the same subject remain similar
    regardless of the occlusion pattern applied. This is the project's
    novel regularization contribution.
    
    Implementation:
        During training, synthetic occlusion is applied to create
        (clean, occluded) pairs from the same base sequence. The loss
        penalizes L2 distance between their embeddings.
    
    Args:
        normalize: Whether to L2-normalize embeddings before comparison.
    """
    
    def __init__(self, normalize: bool = False):
        super().__init__()
        self.normalize = normalize
    
    def forward(
        self,
        clean_embeddings: torch.Tensor,
        occluded_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute consistency loss between clean and occluded embeddings.
        
        Args:
            clean_embeddings: Embeddings from clean sequences (B, D) or (B, P, D).
            occluded_embeddings: Embeddings from synthetically occluded versions (B, D) or (B, P, D).
            
        Returns:
            Tuple of (loss, info_dict).
        """
        if self.normalize:
            clean_embeddings = nn.functional.normalize(clean_embeddings, p=2, dim=-1)
            occluded_embeddings = nn.functional.normalize(occluded_embeddings, p=2, dim=-1)
        
        # L2 distance between paired embeddings
        diff = clean_embeddings - occluded_embeddings
        loss = (diff ** 2).sum(dim=-1).mean()
        
        # If part-based (B, P, D), average across parts
        if loss.dim() > 0:
            loss = loss.mean()
        
        info = {
            'consistency_loss': loss.item(),
            'mean_embedding_dist': diff.norm(dim=-1).mean().item(),
        }
        
        return loss, info
