"""
Model checkpoint management.
Supports saving/loading to Google Drive for Colab persistence.
"""
import os
import torch
from typing import Any, Dict, Optional


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    iteration: int,
    metrics: Dict[str, float],
    config: Dict[str, Any],
    save_dir: str = 'checkpoints',
    filename: Optional[str] = None,
    is_best: bool = False
) -> str:
    """Save a training checkpoint.
    
    Saves model state, optimizer state, scheduler state, iteration count,
    metrics, and config for full experiment resumption.
    
    Args:
        model: The model to save.
        optimizer: Optimizer state.
        scheduler: LR scheduler state.
        iteration: Current iteration number.
        metrics: Current evaluation metrics.
        config: Experiment configuration.
        save_dir: Directory to save checkpoints.
        filename: Optional custom filename.
        is_best: If True, also save as 'best_model.pth'.
        
    Returns:
        Path to the saved checkpoint.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config,
    }
    
    if filename is None:
        filename = f'checkpoint_iter{iteration}.pth'
    
    save_path = os.path.join(save_dir, filename)
    torch.save(checkpoint, save_path)
    print(f"[Checkpoint] Saved: {save_path} (iter={iteration})")
    
    # Save best model separately
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"[Checkpoint] New best model saved: {best_path}")
    
    return save_path


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Any = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Load a training checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file.
        model: Model to load weights into.
        optimizer: Optional optimizer to restore state.
        scheduler: Optional scheduler to restore state.
        device: Device to map tensors to.
        
    Returns:
        Dictionary with 'iteration', 'metrics', and 'config'.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[Checkpoint] Model loaded from: {checkpoint_path}")
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("[Checkpoint] Optimizer state restored")
    
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("[Checkpoint] Scheduler state restored")
    
    info = {
        'iteration': checkpoint.get('iteration', 0),
        'metrics': checkpoint.get('metrics', {}),
        'config': checkpoint.get('config', {}),
    }
    
    print(f"[Checkpoint] Resuming from iteration {info['iteration']}")
    if info['metrics']:
        metrics_str = ' | '.join(f"{k}: {v:.4f}" for k, v in info['metrics'].items())
        print(f"[Checkpoint] Last metrics: {metrics_str}")
    
    return info


def cleanup_checkpoints(save_dir: str = 'checkpoints', keep_last: int = 3) -> None:
    """Remove old checkpoints, keeping only the most recent ones.
    
    Always preserves 'best_model.pth' regardless of keep_last.
    
    Args:
        save_dir: Checkpoint directory.
        keep_last: Number of recent checkpoints to keep.
    """
    if not os.path.exists(save_dir):
        return
    
    checkpoint_files = sorted([
        f for f in os.listdir(save_dir)
        if f.startswith('checkpoint_') and f.endswith('.pth')
    ])
    
    # Remove old checkpoints
    files_to_remove = checkpoint_files[:-keep_last] if len(checkpoint_files) > keep_last else []
    for f in files_to_remove:
        path = os.path.join(save_dir, f)
        os.remove(path)
        print(f"[Checkpoint] Removed old checkpoint: {path}")
