# Occlusion-Robust Gait Recognition (Streamlit Demo)

An interactive **gait recognition demo** that identifies a subject from an occluded walking sequence using an occlusion-aware deep model (**OcclusionGaitNet**).

This app demonstrates how reliability-aware part features improve matching when parts of the body are hidden.

---

## ✨ Highlights

- Interactive UI built with **Streamlit**
- Synthetic gait sequence generation for 10 demo subjects
- Multiple occlusion scenarios:
  - `lower`
  - `upper`
  - `block`
  - `crowd`
- Per-part visibility/reliability prediction
- Top-k identity retrieval with cosine similarity
- Visual diagnostics:
  - Clean vs occluded GEI
  - Per-part reliability bar chart
  - Top-3 gallery matches
  - Occluded input frame strip

---

## 🧠 Model Summary

The demo uses `OcclusionGaitNet`, which includes:

1. **CNN backbone** (3 convolutional blocks)
2. **Occlusion estimator** that predicts reliability scores for 4 horizontal body parts
3. **Part attention** per body region
4. **Reliability-weighted fusion** for robust embedding generation
5. **L2-normalized 128-d embedding** for matching

Body partitions:

1. Head/Shoulders  
2. Torso  
3. Hips/Thighs  
4. Legs/Feet

---

## 🧪 Pipeline (Demo Flow)

```text
Synthetic silhouette frames
        ↓
Compute GEI (temporal average)
        ↓
Apply occlusion (selected type)
        ↓
OcclusionGaitNet
   ├─ Embedding (128-d)
   └─ Part reliability scores (4 parts)
        ↓
Cosine similarity vs gallery embeddings
        ↓
Top-3 identity matches
```

---

## 🖥️ Interface Walkthrough

From the sidebar you can:

- Select a synthetic subject (`Subject 01` … `Subject 10`)
- Select an occlusion type
- Run recognition

After execution, the app shows:

- Rank-1 correctness banner
- Clean vs occluded GEI comparison
- Part reliability plot (with 0.5 threshold guide)
- Top-3 matches and similarity table
- Sample occluded input frames over time

---

## 📦 Requirements

Dependencies are listed in `requirements.txt`:

- streamlit
- torch==2.9.0
- torchvision
- numpy
- opencv-python-headless
- matplotlib
- gdown
- Pillow

---

## 🚀 Setup & Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in your terminal.

---

## 📥 Model Weights

On first run, the app downloads `best_model.pth` from Google Drive using `gdown` and caches it locally.

If loading fails, ensure:

- Internet access is available
- The Google Drive file is public/accessible

---

## 📁 Repository Structure

```text
.
├── app.py            # Full Streamlit app + model definition + visualization logic
├── requirements.txt  # Python dependencies
└── README.md
```

---

## 📌 Notes & Limitations

- This repository is focused on an **interactive demonstration**.
- The gallery and query sequences are synthetic for UI experimentation.
- In-app model metadata references training on CASIA-B with synthetic occlusion augmentation.

---

## 🎓 Project Context

Master’s project: **Deep Learning-Based Gait Recognition Robust to Partial Occlusion**.

