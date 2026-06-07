"""
Logging utilities for experiment tracking.
Supports Weights & Biases and TensorBoard backends.
"""
import os
from typing import Any, Dict, Optional


def setup_logger(config: Dict[str, Any]) -> Any:
    """Initialize experiment logging backend.
    
    Args:
        config: Configuration dictionary with 'logging' section.
        
    Returns:
        Logger object (wandb run or TensorBoard writer).
    """
    log_config = config.get('logging', {})
    backend = log_config.get('backend', 'wandb')
    
    if backend == 'wandb':
        return _setup_wandb(config, log_config)
    elif backend == 'tensorboard':
        return _setup_tensorboard(log_config)
    else:
        print(f"[Logger] Unknown backend '{backend}', using print logging")
        return None


def _setup_wandb(config: Dict[str, Any], log_config: Dict[str, Any]) -> Any:
    """Set up Weights & Biases logging."""
    try:
        import wandb
        run = wandb.init(
            project=log_config.get('project', 'gait-recognition'),
            name=log_config.get('run_name', 'experiment'),
            config=config,
            reinit=True
        )
        print(f"[Logger] W&B initialized: {run.name}")
        return run
    except ImportError:
        print("[Logger] wandb not installed, falling back to print logging")
        return None
    except Exception as e:
        print(f"[Logger] W&B init failed: {e}, falling back to print logging")
        return None


def _setup_tensorboard(log_config: Dict[str, Any]) -> Any:
    """Set up TensorBoard logging."""
    try:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = os.path.join('runs', log_config.get('run_name', 'experiment'))
        writer = SummaryWriter(log_dir=log_dir)
        print(f"[Logger] TensorBoard logging to: {log_dir}")
        return writer
    except ImportError:
        print("[Logger] tensorboard not installed, falling back to print logging")
        return None


def log_metrics(logger: Any, metrics: Dict[str, float], step: int,
                backend: str = 'wandb') -> None:
    """Log metrics to the configured backend.
    
    Args:
        logger: Logger object from setup_logger.
        metrics: Dictionary of metric name -> value.
        step: Current training step/iteration.
        backend: Logging backend type.
    """
    if logger is None:
        # Fallback: print metrics
        metrics_str = ' | '.join(f"{k}: {v:.4f}" for k, v in metrics.items())
        print(f"[Step {step}] {metrics_str}")
        return
    
    if backend == 'wandb':
        import wandb
        wandb.log(metrics, step=step)
    elif backend == 'tensorboard':
        for key, value in metrics.items():
            logger.add_scalar(key, value, step)


def log_images(logger: Any, images: Dict[str, Any], step: int,
               backend: str = 'wandb') -> None:
    """Log images (e.g., attention maps) to the configured backend.
    
    Args:
        logger: Logger object.
        images: Dictionary of image name -> image array (HWC or CHW).
        step: Current training step.
        backend: Logging backend type.
    """
    if logger is None:
        return
    
    if backend == 'wandb':
        import wandb
        wandb_images = {k: wandb.Image(v) for k, v in images.items()}
        wandb.log(wandb_images, step=step)
    elif backend == 'tensorboard':
        for key, img in images.items():
            logger.add_image(key, img, step, dataformats='HWC')


def finish_logging(logger: Any, backend: str = 'wandb') -> None:
    """Clean up and finalize logging.
    
    Args:
        logger: Logger object.
        backend: Logging backend type.
    """
    if logger is None:
        return
    
    if backend == 'wandb':
        import wandb
        wandb.finish()
    elif backend == 'tensorboard':
        logger.close()
    
    print("[Logger] Logging finalized")
