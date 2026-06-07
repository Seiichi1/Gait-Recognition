"""Polished Streamlit demo for occlusion-robust gait recognition."""
from __future__ import annotations

import streamlit as st

from src.demo import infer_identity, make_montage, prepare_sequence, to_uint8


st.set_page_config(
    page_title="Occlusion-Robust Gait Recognition",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=DM+Sans:wght@400;500;700&display=swap');
        :root {
            --bg: #07111f;
            --panel: rgba(10, 25, 47, 0.72);
            --panel-strong: rgba(14, 33, 61, 0.92);
            --border: rgba(126, 211, 255, 0.18);
            --ink: #eaf4ff;
            --muted: #8db2cf;
            --accent: #73f0c4;
            --accent-2: #7ec9ff;
            --accent-3: #ffd37b;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(126, 201, 255, 0.18), transparent 34%),
                radial-gradient(circle at top right, rgba(115, 240, 196, 0.12), transparent 26%),
                linear-gradient(160deg, #030814 0%, #07111f 48%, #0c1830 100%);
            color: var(--ink);
            font-family: 'DM Sans', sans-serif;
        }
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2.5rem;
            max-width: 1240px;
        }
        h1, h2, h3 {
            font-family: 'Space Grotesk', sans-serif;
            letter-spacing: -0.03em;
            color: var(--ink);
        }
        .hero {
            background: linear-gradient(135deg, rgba(8, 20, 39, 0.92), rgba(16, 40, 73, 0.78));
            border: 1px solid var(--border);
            border-radius: 28px;
            padding: 2.4rem 2.5rem;
            box-shadow: 0 24px 80px rgba(2, 10, 24, 0.42);
            overflow: hidden;
        }
        .hero-kicker {
            color: var(--accent);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-weight: 700;
        }
        .hero-title {
            font-size: 3.25rem;
            line-height: 1.02;
            margin: 0.4rem 0 0.9rem 0;
            max-width: 780px;
        }
        .hero-copy {
            color: var(--muted);
            font-size: 1.04rem;
            max-width: 820px;
            line-height: 1.75;
        }
        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.7rem;
            margin-top: 1.25rem;
        }
        .pill {
            padding: 0.55rem 0.9rem;
            border-radius: 999px;
            font-size: 0.86rem;
            background: rgba(115, 240, 196, 0.08);
            border: 1px solid rgba(115, 240, 196, 0.18);
            color: var(--ink);
        }
        .glass-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 1.2rem 1.25rem;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
            margin-top: 1rem;
        }
        .metric-card {
            background: linear-gradient(180deg, rgba(11, 26, 48, 0.96), rgba(9, 20, 36, 0.86));
            border-radius: 22px;
            border: 1px solid var(--border);
            padding: 1rem 1.1rem;
            min-height: 118px;
        }
        .metric-label {
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.74rem;
        }
        .metric-value {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2rem;
            margin-top: 0.45rem;
            color: var(--ink);
        }
        .metric-foot {
            color: #b8d5ea;
            font-size: 0.88rem;
            margin-top: 0.45rem;
        }
        div[data-testid="stMetric"] {
            background: transparent;
            border: none;
        }
        div[data-testid="stImage"] img {
            border-radius: 18px;
            border: 1px solid rgba(126, 211, 255, 0.16);
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.8rem;
        }
        .status-pill {
            border-radius: 18px;
            background: rgba(126, 201, 255, 0.07);
            border: 1px solid rgba(126, 201, 255, 0.18);
            padding: 0.9rem 1rem;
        }
        .status-pill strong {
            display: block;
            margin-bottom: 0.35rem;
            color: var(--ink);
        }
        .caption-soft {
            color: var(--muted);
            font-size: 0.9rem;
        }
        .stButton button {
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
            color: #07111f;
            border-radius: 999px;
            border: none;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            padding: 0.8rem 1.1rem;
            box-shadow: 0 18px 40px rgba(126, 201, 255, 0.22);
        }
        @media (max-width: 900px) {
            .hero-title {
                font-size: 2.45rem;
            }
            .status-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
def render_metric_card(label: str, value: str, foot: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-foot">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_styles()

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Surveillance AI • Research Demonstration</div>
            <div class="hero-title">Occlusion-Robust Human Gait Recognition</div>
            <div class="hero-copy">
                A high-fidelity visual demonstration of a deep-learning pipeline for gait identification under
                carrying, crowd, and static occlusion. The interface presents preprocessing, sequence-level
                representation, retrieval output, and runtime health in one polished control surface.
            </div>
            <div class="pill-row">
                <div class="pill">Occlusion-Aware Feature Fusion</div>
                <div class="pill">GEI / GEnI Analytics</div>
                <div class="pill">Sample Retrieval Pipeline</div>
                <div class="pill">Deployment-Ready Interface</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    control_left, control_right = st.columns([1.1, 1.7], gap="large")

    with control_left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Run Demo")
        occlusion = st.selectbox(
            "Occlusion scenario",
            ["NO", "CA", "CO", "ST"],
            format_func=lambda item: {
                "NO": "Normal Walking",
                "CA": "Carrying Occlusion",
                "CO": "Crowd Occlusion",
                "ST": "Static Occlusion",
            }[item],
        )
        identity_seed = st.slider("Subject profile", min_value=1, max_value=4, value=2)
        trigger = st.button("Execute Recognition Pass", use_container_width=True)
        st.markdown('<div class="caption-soft">Sequence length: 30 frames • Target resolution: 64 × 44</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with control_right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("System Status")
        torch_ready = torch is not None
        status_items = [
            ("Data Pipeline", "Active", "Synthetic sequence generation and silhouette preprocessing are online."),
            ("Feature Encoder", "Ready", "PyTorch encoder engaged." if torch_ready else "Analytical encoder engaged."),
            ("Evaluation Head", "Ready", "Rank-based retrieval and similarity scoring are responding."),
            ("Deployment Surface", "Ready", "Streamlit interface rendered with custom premium theming."),
        ]
        st.markdown('<div class="status-grid">', unsafe_allow_html=True)
        for label, state, text in status_items:
            st.markdown(
                f"""
                <div class="status-pill">
                    <strong>{label} · {state}</strong>
                    <span class="caption-soft">{text}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if "demo_state" not in st.session_state or trigger:
        raw_frames, processed_frames, gei, geni = prepare_sequence(occlusion, identity_seed - 1)
        st.session_state.demo_state = {
            "raw_frames": raw_frames,
            "processed_frames": processed_frames,
            "gei": gei,
            "geni": geni,
            "result": infer_identity(processed_frames),
            "occlusion": occlusion,
        }

    state = st.session_state.demo_state
    raw_frames = state["raw_frames"]
    processed_frames = state["processed_frames"]
    gei = state["gei"]
    geni = state["geni"]
    result = state["result"]

    metrics_cols = st.columns(4, gap="medium")
    with metrics_cols[0]:
        render_metric_card("Predicted Identity", f"#{result['identity']:03d}", f"Best match under {state['occlusion']} occlusion")
    with metrics_cols[1]:
        render_metric_card("Confidence", f"{result['confidence']}%", "Similarity-normalized retrieval confidence")
    with metrics_cols[2]:
        render_metric_card("Rank-1", f"{result['rank1']}%", "Sequence-level top-match performance")
    with metrics_cols[3]:
        render_metric_card("mAP", f"{result['map']}%", result["backend"])

    left, right = st.columns([1.3, 1], gap="large")
    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("View Preprocessing")
        montage = make_montage(processed_frames, every=5)
        st.image(to_uint8(montage), caption="Processed silhouette sequence montage")
        view_cols = st.columns(2, gap="medium")
        with view_cols[0]:
            st.image(to_uint8(gei), caption="Gait Energy Image (GEI)")
        with view_cols[1]:
            st.image(to_uint8(geni), caption="Gait Entropy Image (GEnI)")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("View Results")
        for idx, match in enumerate(result["top_matches"], start=1):
            st.progress(min(match["score"] / 100.0, 1.0), text=f"Top {idx}: {match['label']} · {match['score']}%")
        st.markdown(
            f"""
            <div class="caption-soft" style="margin-top:0.9rem;">
                Rank-5 estimate: <strong style="color:#eaf4ff;">{result['rank5']}%</strong><br/>
                Predicted occlusion affinity: <strong style="color:#eaf4ff;">{result['category']}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Sample Input Footprint")
    sample_cols = st.columns(3, gap="medium")
    sample_indices = [0, len(raw_frames) // 2, len(raw_frames) - 1]
    for col, idx in zip(sample_cols, sample_indices):
        with col:
            st.image(to_uint8(raw_frames[idx]), caption=f"Raw frame {idx + 1:02d}")
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
