# Occlusion-Robust Human Gait Recognition — Google Colab Setup
# ==============================================================
# Run this notebook to set up the complete environment on Google Colab.
# Mount Google Drive, install dependencies, and verify the data pipeline.

# ============================================================
# CELL 1: Mount Google Drive & Clone/Copy Project
# ============================================================
# from google.colab import drive
# drive.mount('/content/drive')
#
# # Option A: Clone from GitHub (if you've pushed the project)
# # !git clone https://github.com/YOUR_USERNAME/gait-recognition.git /content/gait-recognition
#
# # Option B: Copy from Google Drive
# # !cp -r "/content/drive/MyDrive/gait-recognition" /content/gait-recognition
#
# %cd /content/gait-recognition

# ============================================================
# CELL 2: Install Dependencies
# ============================================================
# !pip install -q -r requirements.txt

# ============================================================
# CELL 3: Set Seeds for Reproducibility
# ============================================================
import random
import numpy as np
import torch
import os

SEED = 42

def set_all_seeds(seed):
    """Set all random seeds for reproducibility across Colab sessions."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"All seeds set to {seed}")

set_all_seeds(SEED)

# ============================================================
# CELL 4: Verify GPU
# ============================================================
print("=" * 50)
print("GPU Check")
print("=" * 50)
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"GPU: {gpu_name}")
    print(f"Memory: {gpu_mem:.1f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("WARNING: No GPU detected!")
    print("Enable GPU: Runtime -> Change runtime type -> GPU")
print(f"PyTorch: {torch.__version__}")

# ============================================================
# CELL 5: Verify Project Structure
# ============================================================
import os

expected_files = [
    "requirements.txt",
    "configs/baseline.yaml",
    "configs/occaware.yaml",
    "src/__init__.py",
    "src/data/__init__.py",
    "src/data/dataset.py",
    "src/data/preprocessing.py",
    "src/data/augmentation.py",
    "src/data/sampler.py",
    "src/models/__init__.py",
    "src/losses/__init__.py",
    "src/losses/triplet.py",
    "src/losses/center.py",
    "src/losses/consistency.py",
    "src/evaluation/__init__.py",
    "src/evaluation/metrics.py",
    "src/utils/__init__.py",
    "src/utils/config.py",
    "src/utils/logging_utils.py",
    "src/utils/checkpoint.py",
    "scripts/download_occgait.py",
]

print("=" * 50)
print("Project Structure Verification")
print("=" * 50)
all_ok = True
for f in expected_files:
    exists = os.path.exists(f)
    status = "OK" if exists else "MISSING"
    if not exists:
        all_ok = False
    print(f"  [{status}] {f}")

print()
if all_ok:
    print("All files present!")
else:
    print("WARNING: Some files are missing. Check the project directory.")

# ============================================================
# CELL 6: Test Config Loading
# ============================================================
from src.utils.config import load_config, set_seed

config = load_config("configs/baseline.yaml")
print(f"\nConfig loaded successfully!")
print(f"  Backbone: {config['model']['backbone']}")
print(f"  Feature dim: {config['model']['feature_dim']}")
print(f"  Batch size: {config['training']['batch_size']}")
print(f"  Total iterations: {config['training']['total_iterations']}")
print(f"  LR: {config['training']['optimizer']['lr']}")

# ============================================================
# CELL 7: Test Data Pipeline (with synthetic data)
# ============================================================
from src.data.preprocessing import preprocess_silhouette, compute_gei, compute_geni_fast
from src.data.augmentation import GaitAugmentation

# Create a synthetic silhouette sequence for testing
T, H, W = 30, 128, 88  # Raw dimensions before preprocessing
frames = np.zeros((T, H, W), dtype=np.float32)

# Draw a simple walking figure
for t in range(T):
    # Body
    cy = H // 2
    cx = W // 2
    frames[t, cy-30:cy+30, cx-8:cx+8] = 1.0  # torso
    frames[t, cy-40:cy-30, cx-5:cx+5] = 1.0   # head
    # Legs with phase
    leg_offset = int(5 * np.sin(2 * np.pi * t / T))
    frames[t, cy+30:cy+50, cx-8+leg_offset:cx-2+leg_offset] = 1.0
    frames[t, cy+30:cy+50, cx+2-leg_offset:cx+8-leg_offset] = 1.0

# Test preprocessing
print("\n" + "=" * 50)
print("Data Pipeline Test")
print("=" * 50)

processed_frames = np.stack([
    preprocess_silhouette(f, target_size=(64, 44))
    for f in frames
])
print(f"Preprocessed: {frames.shape} -> {processed_frames.shape}")

# Test GEI
gei = compute_gei(processed_frames)
print(f"GEI shape: {gei.shape}, range: [{gei.min():.3f}, {gei.max():.3f}]")

# Test GEnI
geni = compute_geni_fast(processed_frames)
print(f"GEnI shape: {geni.shape}, range: [{geni.min():.3f}, {geni.max():.3f}]")

# Test augmentation
aug_config = config.get('augmentation', {})
aug = GaitAugmentation(aug_config)
augmented = aug(processed_frames)
print(f"Augmented: {augmented.shape}")

# ============================================================
# CELL 8: Test Loss Functions
# ============================================================
from src.losses.triplet import TripletLoss
from src.losses.center import CenterLoss
from src.losses.consistency import OcclusionConsistencyLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Synthetic embeddings: 8 subjects, 4 sequences each = 32 samples
B = 32
D = 256
num_classes = 8
embeddings = torch.randn(B, D, device=device)
labels = torch.arange(num_classes, device=device).repeat_interleave(4)

print("\n" + "=" * 50)
print("Loss Function Test")
print("=" * 50)

# Triplet loss
triplet_loss = TripletLoss(margin=0.2, mining="hard")
loss_t, info_t = triplet_loss(embeddings, labels)
print(f"Triplet Loss: {loss_t.item():.4f} | Valid triplets: {info_t['num_valid_triplets']}")

# Center loss
center_loss = CenterLoss(num_classes=num_classes, feature_dim=D).to(device)
loss_c, info_c = center_loss(embeddings, labels)
print(f"Center Loss: {loss_c.item():.4f}")

# Consistency loss
consistency_loss = OcclusionConsistencyLoss()
clean = torch.randn(B, D, device=device)
occluded = clean + 0.1 * torch.randn(B, D, device=device)
loss_cons, info_cons = consistency_loss(clean, occluded)
print(f"Consistency Loss: {loss_cons.item():.4f}")

# Total loss
lambda1, lambda2, lambda3 = 1.0, 0.005, 0.1
total_loss = lambda1 * loss_t + lambda2 * loss_c + lambda3 * loss_cons
print(f"Total Loss: {total_loss.item():.4f} (λ1={lambda1}, λ2={lambda2}, λ3={lambda3})")

print("\n" + "=" * 50)
print("ALL TESTS PASSED ✓")
print("=" * 50)
print("\nNext steps:")
print("1. Download the OccGait dataset: python scripts/download_occgait.py")
print("2. Train baseline: python scripts/train_baseline.py --config configs/baseline.yaml")
