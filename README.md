# Gait Recognition

Occlusion-robust human gait recognition project with a Streamlit demo, reusable `src/` packages, training scripts, and a lightweight backend surface.

## Project layout

```text
Gait-Recognition/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── LICENSE
├── src/
├── scripts/
├── notebooks/
├── backend/
└── configs/
```

## Main entrypoints

- `streamlit run app.py` for the demo UI
- `python backend/server.py` for the Flask backend
- `python scripts/train_baseline.py` for baseline training
- `python scripts/train_occaware.py` for occlusion-aware training
- `python scripts/evaluate.py` for evaluation

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

Training and demo defaults live in `configs/`. The current sample configs are:

- `configs/baseline.yaml`
- `configs/occaware.yaml`

## Notes

- The repo keeps the current root `app.py` as the Streamlit entrypoint for compatibility.
- Root-level `src/` modules are organized into `models/`, `data/`, `losses/`, `evaluation/`, and `utils/`.
- Large datasets, checkpoints, and experiment outputs are ignored by `.gitignore`.
