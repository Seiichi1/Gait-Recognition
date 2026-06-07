"""
Configuration management utility.
Loads YAML configs and provides defaults for reproducible experiments.
"""
import yaml
import os
import random
import numpy as np
import torch
from typing import Any, Dict, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.
    
    Args:
        config_path: Path to the YAML config file.
        
    Returns:
        Dictionary containing configuration parameters.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary.
        save_path: Output file path.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility.
    
    Follows Google Colab best practices for consistent runs across sessions.
    Sets seeds for: random, numpy, torch CPU, torch CUDA, and cuDNN.
    
    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic operations for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(config: Dict[str, Any]) -> torch.device:
    """Get the compute device from config.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        torch.device for computation.
    """
    device_str = config.get('hardware', {}).get('device', 'cuda')
    if device_str == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[Config] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Config] GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("[Config] Using CPU")
    return device


def setup_experiment(config_path: str) -> Dict[str, Any]:
    """Full experiment setup: load config, set seeds, get device.
    
    Args:
        config_path: Path to YAML config.
        
    Returns:
        Config dict augmented with 'device' key.
    """
    config = load_config(config_path)
    seed = config.get('seed', 42)
    set_seed(seed)
    print(f"[Config] Seed set to {seed}")
    
    config['device'] = get_device(config)
    
    return config
