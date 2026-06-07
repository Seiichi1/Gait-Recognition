"""
Evaluation metrics for gait recognition.

Supports: Rank-k accuracy, CMC curves, mean Average Precision (mAP).
"""
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


def compute_distance_matrix(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    metric: str = 'euclidean'
) -> np.ndarray:
    """Compute pairwise distance matrix between query and gallery.
    
    Args:
        query_features: (Q, D) feature matrix for probes.
        gallery_features: (G, D) feature matrix for gallery.
        metric: Distance metric ('euclidean' or 'cosine').
        
    Returns:
        (Q, G) distance matrix.
    """
    if metric == 'euclidean':
        # Efficient euclidean distance computation
        q_sq = np.sum(query_features ** 2, axis=1, keepdims=True)  # (Q, 1)
        g_sq = np.sum(gallery_features ** 2, axis=1, keepdims=True)  # (G, 1)
        cross = np.dot(query_features, gallery_features.T)  # (Q, G)
        dist = q_sq - 2 * cross + g_sq.T
        dist = np.maximum(dist, 0)  # Numerical stability
        dist = np.sqrt(dist)
    elif metric == 'cosine':
        q_norm = query_features / (np.linalg.norm(query_features, axis=1, keepdims=True) + 1e-8)
        g_norm = gallery_features / (np.linalg.norm(gallery_features, axis=1, keepdims=True) + 1e-8)
        dist = 1 - np.dot(q_norm, g_norm.T)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return dist


def compute_rank_k(
    dist_matrix: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    k: int = 1,
    query_views: Optional[np.ndarray] = None,
    gallery_views: Optional[np.ndarray] = None,
    exclude_same_view: bool = False
) -> float:
    """Compute Rank-k accuracy.
    
    Args:
        dist_matrix: (Q, G) distance matrix.
        query_labels: (Q,) identity labels for probes.
        gallery_labels: (G,) identity labels for gallery.
        k: Rank position (1 for Rank-1, 5 for Rank-5).
        query_views: Optional (Q,) view labels for probes.
        gallery_views: Optional (G,) view labels for gallery.
        exclude_same_view: If True, exclude gallery samples with same view.
        
    Returns:
        Rank-k accuracy as a float in [0, 1].
    """
    Q = dist_matrix.shape[0]
    correct = 0
    
    for q in range(Q):
        distances = dist_matrix[q].copy()
        
        # Optionally exclude same-view gallery entries
        if exclude_same_view and query_views is not None and gallery_views is not None:
            same_view = gallery_views == query_views[q]
            distances[same_view] = float('inf')
        
        # Get k nearest neighbors
        top_k_indices = np.argsort(distances)[:k]
        top_k_labels = gallery_labels[top_k_indices]
        
        if query_labels[q] in top_k_labels:
            correct += 1
    
    return correct / Q


def compute_cmc(
    dist_matrix: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    max_rank: int = 50,
    query_views: Optional[np.ndarray] = None,
    gallery_views: Optional[np.ndarray] = None,
    exclude_same_view: bool = False
) -> np.ndarray:
    """Compute Cumulative Match Characteristic (CMC) curve.
    
    Args:
        dist_matrix: (Q, G) distance matrix.
        query_labels: (Q,) identity labels.
        gallery_labels: (G,) identity labels.
        max_rank: Maximum rank to compute.
        query_views: Optional view labels for probes.
        gallery_views: Optional view labels for gallery.
        exclude_same_view: If True, exclude same-view gallery entries.
        
    Returns:
        CMC curve of shape (max_rank,) — cumulative accuracy at each rank.
    """
    cmc = np.zeros(max_rank)
    
    for k in range(1, max_rank + 1):
        cmc[k - 1] = compute_rank_k(
            dist_matrix, query_labels, gallery_labels, k,
            query_views, gallery_views, exclude_same_view
        )
    
    return cmc


def compute_map(
    dist_matrix: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    query_views: Optional[np.ndarray] = None,
    gallery_views: Optional[np.ndarray] = None,
    exclude_same_view: bool = False
) -> float:
    """Compute mean Average Precision (mAP).
    
    Args:
        dist_matrix: (Q, G) distance matrix.
        query_labels: (Q,) identity labels.
        gallery_labels: (G,) identity labels.
        query_views: Optional view labels for probes.
        gallery_views: Optional view labels for gallery.
        exclude_same_view: If True, exclude same-view gallery entries.
        
    Returns:
        mAP score in [0, 1].
    """
    Q = dist_matrix.shape[0]
    aps = []
    
    for q in range(Q):
        distances = dist_matrix[q].copy()
        
        if exclude_same_view and query_views is not None and gallery_views is not None:
            valid = gallery_views != query_views[q]
            distances[~valid] = float('inf')
        
        # Sort by distance
        sorted_indices = np.argsort(distances)
        sorted_labels = gallery_labels[sorted_indices]
        
        # Compute AP for this query
        relevant = sorted_labels == query_labels[q]
        if relevant.sum() == 0:
            continue
        
        cumsum = np.cumsum(relevant)
        precision_at_k = cumsum / (np.arange(len(relevant)) + 1)
        ap = (precision_at_k * relevant).sum() / relevant.sum()
        aps.append(ap)
    
    return np.mean(aps) if aps else 0.0


def evaluate_all(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    query_views: Optional[np.ndarray] = None,
    gallery_views: Optional[np.ndarray] = None,
    query_categories: Optional[np.ndarray] = None,
    metric: str = 'euclidean',
    exclude_same_view: bool = False
) -> Dict[str, float]:
    """Run complete evaluation: Rank-1, Rank-5, CMC AUC, mAP.
    
    Also computes per-category breakdown if categories are provided.
    
    Args:
        query_features: (Q, D) probe feature matrix.
        gallery_features: (G, D) gallery feature matrix.
        query_labels: (Q,) probe identity labels.
        gallery_labels: (G,) gallery identity labels.
        query_views: Optional (Q,) probe view labels.
        gallery_views: Optional (G,) gallery view labels.
        query_categories: Optional (Q,) occlusion category labels.
        metric: Distance metric.
        exclude_same_view: Whether to exclude same-view gallery items.
        
    Returns:
        Dictionary of metric name -> value.
    """
    dist_matrix = compute_distance_matrix(query_features, gallery_features, metric)
    
    results = {}
    
    # Overall metrics
    results['rank1'] = compute_rank_k(
        dist_matrix, query_labels, gallery_labels, 1,
        query_views, gallery_views, exclude_same_view
    )
    results['rank5'] = compute_rank_k(
        dist_matrix, query_labels, gallery_labels, 5,
        query_views, gallery_views, exclude_same_view
    )
    
    cmc = compute_cmc(
        dist_matrix, query_labels, gallery_labels, 50,
        query_views, gallery_views, exclude_same_view
    )
    results['cmc_auc'] = np.trapz(cmc, dx=1.0 / len(cmc))
    
    results['map'] = compute_map(
        dist_matrix, query_labels, gallery_labels,
        query_views, gallery_views, exclude_same_view
    )
    
    # Per-category breakdown
    if query_categories is not None:
        categories = np.unique(query_categories)
        for cat in categories:
            cat_mask = query_categories == cat
            cat_dist = dist_matrix[cat_mask]
            cat_labels = query_labels[cat_mask]
            cat_views = query_views[cat_mask] if query_views is not None else None
            
            results[f'rank1_{cat}'] = compute_rank_k(
                cat_dist, cat_labels, gallery_labels, 1,
                cat_views, gallery_views, exclude_same_view
            )
            results[f'rank5_{cat}'] = compute_rank_k(
                cat_dist, cat_labels, gallery_labels, 5,
                cat_views, gallery_views, exclude_same_view
            )
    
    return results


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract feature embeddings from a model for all samples in dataloader.
    
    Args:
        model: Trained model in eval mode.
        dataloader: DataLoader to extract features from.
        device: Compute device.
        
    Returns:
        Tuple of (features, labels, views, categories) as numpy arrays.
    """
    model.eval()
    
    all_features = []
    all_labels = []
    all_views = []
    all_categories = []
    
    for batch in dataloader:
        silhouettes = batch['silhouettes'].to(device)
        
        # Forward pass to get embeddings
        embeddings = model(silhouettes)
        
        # Flatten if part-based (B, P, D) -> (B, P*D)
        if embeddings.dim() == 3:
            embeddings = embeddings.reshape(embeddings.size(0), -1)
        
        all_features.append(embeddings.cpu().numpy())
        all_labels.append(batch['label'].numpy())
        all_views.extend(batch['view'])
        all_categories.extend(batch['category'])
    
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    views = np.array(all_views)
    categories = np.array(all_categories)
    
    return features, labels, views, categories
