"""
Occlusion-Robust Gait Recognition — Interactive Demo
Deployed via Streamlit Cloud
"""

import os
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import gdown
import json

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Occlusion-Robust Gait Recognition",
    page_icon="🚶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_FILE_ID = "1-wNtOAe3pHEDvLubvkF22meNFrAyMBK5"
MODEL_PATH    = "best_model.pth"
IMG_H, IMG_W  = 64, 44
EMBED_DIM     = 128
N_PARTS       = 4
OCC_RATIO     = 0.4
DEVICE        = torch.device("cpu")  # Streamlit Cloud has no GPU

PART_LABELS   = ["Head/Shoulders", "Torso", "Hips/Thighs", "Legs/Feet"]
OCC_COLORS    = {
    "lower": "#3498db",
    "upper": "#e74c3c",
    "block": "#f39c12",
    "crowd": "#9b59b6"
}
OCC_DESCRIPTIONS = {
    "lower": "Lower-body occlusion — simulates a wall, fence, or parked vehicle blocking the lower 40% of the frame.",
    "upper": "Upper-body occlusion — simulates a passing vehicle or overhead obstruction blocking the upper 35%.",
    "block": "Block occlusion — simulates a static obstacle such as a pillar or bollard.",
    "crowd": "Crowd occlusion — simulates pedestrians walking in front, creating vertical strip occlusions."
}

# ── Model definition (inline to avoid import issues on Streamlit Cloud) ───────
import torch.nn as nn

class PartAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4), nn.ReLU(),
            nn.Linear(channels // 4, channels), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x)

class OcclusionGaitNet(nn.Module):
    def __init__(self, embed_dim=128, n_parts=4):
        super().__init__()
        self.n_parts = n_parts
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.occ_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_parts), nn.Sigmoid()
        )
        self.part_attention = nn.ModuleList([PartAttention(128) for _ in range(n_parts)])
        self.projector = nn.Sequential(
            nn.Linear(128 * n_parts, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, embed_dim)
        )

    def extract_parts(self, feat_map):
        B, C, H, W = feat_map.shape
        strip_h = H // self.n_parts
        return [feat_map[:, :, i*strip_h:(i+1)*strip_h, :].mean(dim=[2,3])
                for i in range(self.n_parts)]

    def forward(self, x, return_reliability=False):
        feat        = self.backbone(x)
        reliability = self.occ_estimator(feat)
        parts       = self.extract_parts(feat)
        weighted    = [self.part_attention[i](p) * reliability[:, i:i+1]
                       for i, p in enumerate(parts)]
        embed = F.normalize(self.projector(torch.cat(weighted, dim=1)), dim=1)
        return (embed, reliability) if return_reliability else embed


# ── Utility functions ─────────────────────────────────────────────────────────
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights (~15MB)..."):
            url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=True)
    return os.path.exists(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_model():
    if not download_model():
        return None
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
    model = OcclusionGaitNet(embed_dim=EMBED_DIM, n_parts=N_PARTS).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def generate_synthetic_subject(subject_id, n_frames=20, h=IMG_H, w=IMG_W):
    """
    Generate a synthetic walking silhouette sequence.
    Each subject has a unique body shape so embeddings differ.
    """
    np.random.seed(subject_id * 42)
    frames = []
    # Body proportions unique per subject
    head_r   = int(4 + subject_id % 3)
    torso_h  = int(20 + subject_id % 8)
    torso_w  = int(10 + subject_id % 6)
    leg_h    = int(18 + subject_id % 5)

    for t in range(n_frames):
        frame = np.zeros((h, w), dtype=np.uint8)
        cx    = w // 2
        phase = (t / n_frames) * 2 * np.pi

        # Head
        cy_head = 6
        cv2.circle(frame, (cx, cy_head), head_r, 255, -1)

        # Torso
        tx1 = cx - torso_w // 2
        ty1 = cy_head + head_r
        cv2.rectangle(frame, (tx1, ty1), (tx1 + torso_w, ty1 + torso_h), 255, -1)

        # Arms (swing with phase)
        arm_swing = int(4 * np.sin(phase))
        cv2.line(frame, (tx1, ty1 + 4),
                 (tx1 - 5, ty1 + torso_h // 2 + arm_swing), 255, 3)
        cv2.line(frame, (tx1 + torso_w, ty1 + 4),
                 (tx1 + torso_w + 5, ty1 + torso_h // 2 - arm_swing), 255, 3)

        # Legs (walking cycle)
        hip_y  = ty1 + torso_h
        l_swing = int(8 * np.sin(phase))
        r_swing = int(8 * np.sin(phase + np.pi))
        cv2.line(frame, (cx - 3, hip_y),
                 (cx - 3 + l_swing, hip_y + leg_h), 255, 4)
        cv2.line(frame, (cx + 3, hip_y),
                 (cx + 3 + r_swing, hip_y + leg_h), 255, 4)

        frames.append(frame)
    return frames


def compute_gei(frames):
    return np.stack(frames).astype(np.float32).mean(axis=0) / 255.0


def apply_occlusion(frames, occ_type, ratio=OCC_RATIO):
    result = []
    for f in frames:
        f = f.copy()
        H, W = f.shape
        if occ_type == "lower":
            f[int(H*(1-ratio)):, :] = 0
        elif occ_type == "upper":
            f[:int(H*ratio), :] = 0
        elif occ_type == "block":
            r, c = int(H*ratio), int(W*ratio)
            f[H//4:H//4+r, W//4:W//4+c] = 0
        elif occ_type == "crowd":
            sw = int(W*ratio*0.5)
            f[:, W//4:W//4+sw] = 0
            f[:, W//2:W//2+sw] = 0
        result.append(f)
    return result


def gei_to_tensor(gei):
    return torch.from_numpy(gei).unsqueeze(0).unsqueeze(0).float().to(DEVICE)


@st.cache_data(show_spinner=False)
def build_gallery(_model, n_subjects=10):
    """Pre-compute gallery embeddings for synthetic subjects."""
    gallery = []
    for sid in range(1, n_subjects + 1):
        frames = generate_synthetic_subject(sid)
        gei    = compute_gei(frames)
        with torch.no_grad():
            emb = _model(gei_to_tensor(gei)).squeeze().cpu().numpy()
        gallery.append({
            "id":    f"Subject {sid:02d}",
            "embed": emb,
            "gei":   gei
        })
    return gallery


def get_top_matches(query_embed, gallery, top_k=3):
    q = torch.tensor(query_embed).unsqueeze(0)
    scores = []
    for g in gallery:
        g_emb = torch.tensor(g["embed"]).unsqueeze(0)
        sim   = F.cosine_similarity(q, g_emb).item()
        scores.append((g["id"], sim, g["gei"]))
    scores.sort(key=lambda x: -x[1])
    return scores[:top_k]


def plot_reliability(reliability, occ_type):
    fig, ax = plt.subplots(figsize=(4, 2.5))
    colors  = [OCC_COLORS.get(occ_type, "#3498db")] * N_PARTS
    bars    = ax.bar(PART_LABELS, reliability, color=colors,
                     edgecolor="black", linewidth=0.8, alpha=0.85)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1,
               label="Threshold (0.5)")
    for bar, val in zip(bars, reliability):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Reliability Score")
    ax.set_title("Per-Part Visibility Estimate", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7)
    plt.xticks(fontsize=7, rotation=10)
    plt.tight_layout()
    return fig


def plot_gei_comparison(clean_gei, occ_gei, occ_type):
    fig, axes = plt.subplots(1, 2, figsize=(5, 3))
    axes[0].imshow(clean_gei, cmap="hot", vmin=0, vmax=1)
    axes[0].set_title("Clean GEI", fontsize=9, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(occ_gei, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title(f"Occluded GEI\n({occ_type})", fontsize=9, fontweight="bold")
    axes[1].axis("off")
    plt.tight_layout()
    return fig


def plot_top_matches(matches, query_label):
    n = len(matches)
    fig, axes = plt.subplots(1, n, figsize=(3*n, 3))
    if n == 1:
        axes = [axes]
    for i, (label, sim, gei) in enumerate(matches):
        axes[i].imshow(gei, cmap="gray", vmin=0, vmax=1)
        rank_color = "#2ecc71" if i == 0 else "#95a5a6"
        axes[i].set_title(f"Rank {i+1}: {label}\nSim: {sim:.3f}",
                          fontsize=8, fontweight="bold", color=rank_color)
        axes[i].axis("off")
        for spine in axes[i].spines.values():
            spine.set_edgecolor(rank_color)
            spine.set_linewidth(3)
    fig.suptitle(f"Top-{n} Gallery Matches for {query_label}",
                 fontsize=9, fontweight="bold")
    plt.tight_layout()
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.title("🚶 Gait Recognition")
    st.markdown("**Occlusion-Robust Identification**")
    st.divider()

    st.subheader("Select Subject")
    subject_id = st.selectbox(
        "Pre-loaded synthetic subject",
        options=list(range(1, 11)),
        format_func=lambda x: f"Subject {x:02d}"
    )

    st.subheader("Occlusion Type")
    occ_type = st.radio(
        "Select occlusion scenario",
        options=["lower", "upper", "block", "crowd"],
        format_func=lambda x: x.capitalize(),
        index=0
    )
    st.caption(OCC_DESCRIPTIONS[occ_type])

    st.divider()
    st.subheader("Model Info")
    st.markdown("""
    - **Architecture:** OcclusionGaitNet
    - **Backbone:** 3-block CNN
    - **Parts:** 4 horizontal strips
    - **Embedding:** 128-dim L2-normalised
    - **Training:** 33 epochs, CASIA-B
    - **Best Rank-1:** 15.1% (occluded)
    - **Dataset:** CASIA-B + synthetic occlusion
    """)

    st.divider()
    run_btn = st.button("▶ Run Recognition", type="primary", use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN PAGE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.title("Deep Learning-Based Gait Recognition Robust to Partial Occlusion")
st.markdown(
    "A demonstration of occlusion-aware gait identification using hierarchical "
    "part attention and reliability-weighted feature fusion. "
    "Trained on CASIA-B silhouettes with synthetic occlusion augmentation."
)

# Load model
with st.spinner("Loading model..."):
    model = load_model()

if model is None:
    st.error("Failed to load model weights. Check that the Drive link is public.")
    st.stop()

# Build gallery
with st.spinner("Building gallery..."):
    gallery = build_gallery(model)

st.success(f"Model loaded — gallery of {len(gallery)} subjects ready.")
st.divider()

# ── Default state: show the selected subject's clean GEI ─────────────────────
frames    = generate_synthetic_subject(subject_id)
clean_gei = compute_gei(frames)

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader(f"Subject {subject_id:02d} — Clean GEI")
    fig_clean, ax = plt.subplots(figsize=(2.5, 3.5))
    ax.imshow(clean_gei, cmap="hot", vmin=0, vmax=1)
    ax.axis("off")
    ax.set_title("Gait Energy Image", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig_clean, use_container_width=False)
    plt.close()

with col2:
    st.subheader("What is a GEI?")
    st.markdown("""
    A **Gait Energy Image (GEI)** is computed by averaging binary silhouette
    frames over a complete gait cycle. Brighter regions indicate where the body
    consistently appears across frames — capturing the characteristic motion
    pattern of an individual's walk.

    **Pipeline:**
    1. Extract silhouette frames from surveillance footage
    2. Resize and binarise each frame (64×44 px)
    3. Compute temporal mean → GEI
    4. Apply synthetic occlusion to simulate real-world obstructions
    5. Feed occluded GEI into OcclusionGaitNet
    6. Match against gallery using cosine similarity on 128-dim embeddings
    """)

st.divider()

# ── Run recognition ───────────────────────────────────────────────────────────
if run_btn:
    occ_frames = apply_occlusion(frames, occ_type)
    occ_gei    = compute_gei(occ_frames)

    with torch.no_grad():
        occ_tensor = gei_to_tensor(occ_gei)
        embed, reliability = model(occ_tensor, return_reliability=True)
        embed_np = embed.squeeze().cpu().numpy()
        rel_np   = reliability.squeeze().cpu().numpy()

    matches = get_top_matches(embed_np, gallery)
    correct = matches[0][0] == f"Subject {subject_id:02d}"

    # ── Result banner ──────────────────────────────────────────────────────
    if correct:
        st.success(f"✅ Correct identification — Subject {subject_id:02d} matched at Rank-1")
    else:
        st.warning(
            f"❌ Incorrect at Rank-1 — predicted {matches[0][0]}, "
            f"true identity is Subject {subject_id:02d}"
        )

    st.divider()

    # ── Row 1: GEI comparison + reliability ───────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("GEI: Clean vs Occluded")
        fig_comp = plot_gei_comparison(clean_gei, occ_gei, occ_type)
        st.pyplot(fig_comp, use_container_width=True)
        plt.close()

    with col_b:
        st.subheader("Part Reliability Estimate")
        st.caption(
            "The model estimates how visible each body part is. "
            "Parts scoring below 0.5 (red dashed line) are considered unreliable "
            "and are down-weighted in the final embedding."
        )
        fig_rel = plot_reliability(rel_np, occ_type)
        st.pyplot(fig_rel, use_container_width=True)
        plt.close()

    st.divider()

    # ── Row 2: Top matches ─────────────────────────────────────────────────
    st.subheader("Top-3 Gallery Matches")
    fig_matches = plot_top_matches(matches, f"Subject {subject_id:02d}")
    st.pyplot(fig_matches, use_container_width=True)
    plt.close()

    # Match scores table
    st.caption("Cosine similarity scores (higher = more similar)")
    match_data = {
        "Rank":       [1, 2, 3],
        "Identity":   [m[0] for m in matches],
        "Similarity": [f"{m[1]:.4f}" for m in matches],
        "Correct":    ["✅" if m[0] == f"Subject {subject_id:02d}" else "❌"
                       for m in matches]
    }
    st.table(match_data)

    st.divider()

    # ── Row 3: Frame strip ─────────────────────────────────────────────────
    st.subheader("Input Sequence — Occluded Frames")
    n_show  = 8
    indices = np.linspace(0, len(occ_frames)-1, n_show, dtype=int)
    cols    = st.columns(n_show)
    for i, (idx, col) in enumerate(zip(indices, cols)):
        with col:
            fig_f, ax_f = plt.subplots(figsize=(1.2, 1.8))
            ax_f.imshow(occ_frames[idx], cmap="gray", vmin=0, vmax=255)
            ax_f.axis("off")
            ax_f.set_title(f"t={idx}", fontsize=7)
            plt.tight_layout()
            st.pyplot(fig_f, use_container_width=True)
            plt.close()

else:
    # ── Placeholder before run ─────────────────────────────────────────────
    st.info(
        "👈 Select a subject and occlusion type from the sidebar, "
        "then click **▶ Run Recognition** to see the model in action."
    )

    # Show occlusion type preview
    st.subheader("Occlusion Type Preview")
    st.caption(f"Currently selected: **{occ_type.capitalize()}** — {OCC_DESCRIPTIONS[occ_type]}")

    preview_frames = generate_synthetic_subject(subject_id)
    preview_occ    = apply_occlusion(preview_frames, occ_type)
    preview_clean  = compute_gei(preview_frames)
    preview_occ_g  = compute_gei(preview_occ)

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        fig_p1, ax1 = plt.subplots(figsize=(2, 3))
        ax1.imshow(preview_clean, cmap="hot", vmin=0, vmax=1)
        ax1.set_title("Clean", fontsize=9); ax1.axis("off")
        plt.tight_layout(); st.pyplot(fig_p1); plt.close()
    with c2:
        fig_p2, ax2 = plt.subplots(figsize=(2, 3))
        ax2.imshow(preview_occ_g, cmap="hot", vmin=0, vmax=1)
        ax2.set_title(f"{occ_type.capitalize()}\nOcclusion", fontsize=9)
        ax2.axis("off"); plt.tight_layout(); st.pyplot(fig_p2); plt.close()
    with c3:
        st.markdown(f"### {occ_type.capitalize()} Occlusion")
        st.markdown(OCC_DESCRIPTIONS[occ_type])
        st.markdown(f"**Occlusion ratio:** {OCC_RATIO} ({int(OCC_RATIO*100)}% of frame)")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Master's Project — Deep Learning-Based Gait Recognition Robust to Partial Occlusion | "
    "Trained on CASIA-B (124 subjects, 11 views) with synthetic occlusion augmentation | "
    "Model: OcclusionGaitNet with hierarchical part attention and reliability-weighted fusion"
)
