import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Celestial Object Classifier", layout="wide", page_icon="🌌")

st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Trebuchet MS', 'Segoe UI', sans-serif;
    }

    .stApp {
        background: radial-gradient(ellipse at 20% 50%, #0d1b3e 0%, #020b18 60%, #000000 100%);
        color: #e0e8ff;
    }


    h1, h2, h3 {
        font-family: 'Courier New', 'Lucida Console', monospace !important;
        letter-spacing: 0.05em;
    }

    .main-title {
        font-family: 'Courier New', 'Lucida Console', monospace;
        font-size: 2.4rem;
        font-weight: 900;
        background: linear-gradient(135deg, #7eb8ff 0%, #a78bfa 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: 0.1em;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 0.95rem;
        color: #6b8ccc;
        letter-spacing: 0.05em;
        margin-bottom: 2rem;
    }

    .section-header {
        font-family: 'Courier New', 'Lucida Console', monospace;
        font-size: 0.8rem;
        font-weight: 600;
        color: #38bdf8;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        border-bottom: 1px solid rgba(56, 189, 248, 0.3);
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    .stSlider > div > div > div {
        background: linear-gradient(90deg, #38bdf8, #7c3aed) !important;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #1e3a6e, #2d1b69);
        color: #7eb8ff;
        font-family: 'Courier New', 'Lucida Console', monospace;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.15em;
        padding: 0.75rem 1.5rem;
        border: 1px solid rgba(126, 184, 255, 0.4);
        border-radius: 4px;
        transition: all 0.3s ease;
        text-transform: uppercase;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #2d4a8e, #3d2b79);
        border-color: rgba(126, 184, 255, 0.8);
        color: #ffffff;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.3);
    }

    .result-galaxy {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.05));
        border: 1px solid rgba(16, 185, 129, 0.5);
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        font-family: 'Courier New', 'Lucida Console', monospace;
        font-size: 1.4rem;
        font-weight: 700;
        color: #34d399;
        letter-spacing: 0.1em;
    }

    .result-star {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.15), rgba(245, 158, 11, 0.05));
        border: 1px solid rgba(251, 191, 36, 0.5);
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        font-family: 'Courier New', 'Lucida Console', monospace;
        font-size: 1.4rem;
        font-weight: 700;
        color: #fbbf24;
        letter-spacing: 0.1em;
    }

    .result-qso {
        background: linear-gradient(135deg, rgba(167, 139, 250, 0.15), rgba(139, 92, 246, 0.05));
        border: 1px solid rgba(167, 139, 250, 0.5);
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        font-family: 'Courier New', 'Lucida Console', monospace;
        font-size: 1.4rem;
        font-weight: 700;
        color: #a78bfa;
        letter-spacing: 0.1em;
    }

    .info-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(126, 184, 255, 0.15);
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }

    .info-card-title {
        font-family: 'Courier New', 'Lucida Console', monospace;
        font-size: 0.7rem;
        color: #38bdf8;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }

    .info-card-value {
        font-size: 0.95rem;
        color: #c8d8f0;
    }

    .divider {
        border: none;
        border-top: 1px solid rgba(56, 189, 248, 0.2);
        margin: 1.5rem 0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(5, 15, 40, 0.95) !important;
        border-right: 1px solid rgba(56, 189, 248, 0.2);
    }

    [data-testid="stSidebar"] * {
        color: #c8d8f0 !important;
    }

    .stSelectbox > div > div {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(126, 184, 255, 0.3) !important;
        color: #e0e8ff !important;
    }

    /* Warning for invalid values */
    .warning-box {
        background: rgba(251, 191, 36, 0.1);
        border: 1px solid rgba(251, 191, 36, 0.4);
        border-radius: 6px;
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
        color: #fbbf24;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

CLASS_META = {
    "GALAXY": {
        "icon": "🌌",
        "css": "result-galaxy",
        "label": "GALAXY",
        "desc": "A vast system of stars, gas, and dark matter bound by gravity.",
        "redshift_range": "0.01 – 2.0",
        "color": "#34d399",
    },
    "STAR": {
        "icon": "⭐",
        "css": "result-star",
        "label": "STAR",
        "desc": "A luminous sphere of plasma held together by its own gravity.",
        "redshift_range": "≈ 0 (−0.01 to 0.01)",
        "color": "#fbbf24",
    },
    "QSO": {
        "icon": "✨",
        "css": "result-qso",
        "label": "QUASAR",
        "desc": "An extremely luminous active galactic nucleus powered by a supermassive black hole.",
        "redshift_range": "0.1 – 7.0+",
        "color": "#a78bfa",
    },
}

BAND_MEDIANS = {"u": 22.18, "g": 21.10, "r": 20.13, "i": 19.41, "z": 19.00}

def clean_band(val, median):
    return median if val < 0 else val

@st.cache_resource
def loadmodel():
    try:
        return joblib.load("star_model.pkl")
    except FileNotFoundError:
        return None



st.markdown('<div class="main-title">🌌 CELESTIAL CLASSIFIER</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">SDSS Photometric Object Classification</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

pipeline = loadmodel()

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="section-header">📡 Photometric Bands (Apparent Magnitude)</div>', unsafe_allow_html=True)

    u = st.slider("u : Ultraviolet", 14.0, 35.0, 22.18, 0.01)
    g = st.slider("g : Green", 14.0, 35.0, 21.10, 0.01)
    r = st.slider("r : Red", 14.0,  35.0, 20.13, 0.01)
    i = st.slider("i : Near-Infrared", 14.0, 35.0, 19.41, 0.01)
    z = st.slider("z : Infrared", 14.0, 35.0, 19.00, 0.01)

with col_right:
    st.markdown('<div class="section-header">🔭 Spectroscopic Data</div>', unsafe_allow_html=True)

    redshift = st.number_input(
        "Redshift",
        min_value=-0.1, max_value=8.0, value=0.42, step=0.001, format="%.6f",
    )

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📊 Redshift Reference Guide</div>', unsafe_allow_html=True)

    fig_ref = go.Figure()
    categories = ["STAR", "GALAXY", "QSO"]
    z_low  = [0,    0.01,  0.1]
    z_high = [0.01, 2.0,   7.0]
    colors = ["#fbbf24", "#34d399", "#a78bfa"]

    for cat, zl, zh, col in zip(categories, z_low, z_high, colors):
        fig_ref.add_trace(go.Bar(
            name=cat, x=[cat], y=[zh - zl],
            base=zl,
            marker_color=col,
            marker_line_width=0,
            width=0.4,
        ))

    fig_ref.add_hline(
        y=redshift, line_dash="dot",
        line_color="white", line_width=1.5,
        annotation_text=f"  z = {redshift:.4f}",
        annotation_font_color="white",
        annotation_font_size=11,
    )

    st.plotly_chart(fig_ref, use_container_width=True)

invalid_bands = [name for name, val in zip(['u','g','z'], [u, g, z]) if val < 0]

st.markdown('<hr class="divider">', unsafe_allow_html=True)

if st.button("🔭  CLASSIFY OBJECT"):

    u_c = clean_band(u, BAND_MEDIANS["u"])
    g_c = clean_band(g, BAND_MEDIANS["g"])
    r_c = clean_band(r, BAND_MEDIANS["r"])
    i_c = clean_band(i, BAND_MEDIANS["i"])
    z_c = clean_band(z, BAND_MEDIANS["z"])

    input_df = pd.DataFrame([{
        "u": u_c, "g": g_c, "r": r_c,
        "i": i_c, "z": z_c, "redshift": redshift
    }])

    features   = pipeline["features"]
    scaled     = pipeline["scaler"].transform(input_df[features])
    pred_enc   = pipeline["model"].predict(scaled)[0]
    pred_label = pipeline["label_encoder"].inverse_transform([pred_enc])[0]
    proba      = pipeline["model"].predict_proba(scaled)[0]
    classes    = pipeline["label_encoder"].classes_

    meta = CLASS_META[pred_label]
    confidence = max(proba) * 100
    proba_dict = dict(zip(classes, proba))

    res_col1, res_col2, res_col3 = st.columns([2, 1, 1])

    with res_col1:
        st.markdown(
            f'<div class="{meta["css"]}">'
            f'{meta["icon"]} {meta["label"]}'
            f'</div>',
            unsafe_allow_html=True
        )
        st.caption(meta["desc"])

    with res_col2:
        st.metric("Confidence", f"{confidence:.2f}%")
        st.metric("Redshift", f"{redshift:.6f}")

    with res_col3:
        risk_map = {"GALAXY": "Extragalactic", "STAR": "Milky Way", "QSO": "Extragalactic"}
        st.metric("Origin", risk_map[pred_label])
        st.metric("Typical z range", meta["redshift_range"])

    st.markdown('<br>', unsafe_allow_html=True)
    bar_colors = [CLASS_META[c]["color"] for c in classes]

    fig_prob = go.Figure(data=[
        go.Bar(
            x=list(classes),
            y=[proba_dict.get(c, 0) for c in classes],
            marker_color=bar_colors,
            marker_line_width=0,
            text=[f"{proba_dict.get(c,0)*100:.1f}%" for c in classes],
            textposition='outside',
            textfont=dict(color='#c8d8f0', size=13, family='Courier New'),
            width=0.4,
        )
    ])

    fig_prob.update_layout(
        title=dict(text="Classification Probabilities", font=dict(family='Courier New', size=13, color='#7eb8ff')),
        yaxis=dict(
            tickformat=".0%", range=[0, 1.15],
            gridcolor='rgba(255,255,255,0.07)', color='#6b8ccc',
            title="Probability",
        ),
        xaxis=dict(showgrid=False, color='#c8d8f0', tickfont=dict(family='Courier New', size=11)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#c8d8f0'),
        height=320,
        margin=dict(t=50, b=40, l=60, r=20),
        showlegend=False,
    )
    st.plotly_chart(fig_prob, use_container_width=True)


with st.sidebar:
    st.markdown("### 🌌 About the Classes")

    for cls, meta in CLASS_META.items():
        st.markdown(f"""
        <div class="info-card">
            <div class="info-card-title">{meta['icon']} {meta['label']}</div>
            <div class="info-card-value">{meta['desc']}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("""**Redshift** is the most predictive feature as it measures how fast an object moves away from us.""")
