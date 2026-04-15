"""
🔥 Forest Fire Prediction System
Phase 2 — Streamlit Frontend
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Forest Fire Predictor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme CSS injection ──────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root palette ── */
:root {
    --bg:       #0b0d14;
    --surface:  #13151f;
    --card:     #1a1d2e;
    --border:   #252840;
    --accent:   #ff6b35;
    --accent2:  #ff3b3b;
    --blue:     #4f8ff7;
    --green:    #22c55e;
    --text:     #e8e8f0;
    --muted:    #7878a0;
    --font-h:   'Syne', sans-serif;
    --font-b:   'DM Sans', sans-serif;
}

/* ── Global reset ── */
html, body, [class*="css"]           { font-family: var(--font-b); color: var(--text); }
.stApp                               { background: var(--bg); }
section[data-testid="stSidebar"]     { background: var(--surface); border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] *   { color: var(--text) !important; }

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header            { visibility: hidden; }

/* ── Headings ── */
h1, h2, h3                           { font-family: var(--font-h) !important; letter-spacing: -0.02em; }

/* ── Sidebar nav radio ── */
div[data-testid="stRadio"] label     { font-size: 0.95rem; padding: 6px 0; cursor: pointer; }

/* ── Metric cards ── */
[data-testid="stMetric"]             { background: var(--card); border: 1px solid var(--border);
                                       border-radius: 12px; padding: 1rem 1.2rem; }
[data-testid="stMetricLabel"]        { color: var(--muted) !important; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .08em; }
[data-testid="stMetricValue"]        { color: var(--text) !important; font-family: var(--font-h) !important; }

/* ── Buttons ── */
.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--font-h) !important;
    font-weight: 700 !important;
    letter-spacing: .04em !important;
    padding: .55rem 2rem !important;
    transition: opacity .15s, transform .1s;
}
.stButton > button:hover { opacity: .88; transform: translateY(-1px); }

/* ── Selectbox / number inputs ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

/* ── DataFrames ── */
.stDataFrame { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }

/* ── Custom card helper ── */
.ff-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.ff-tag {
    display: inline-block;
    background: rgba(255,107,53,.15);
    color: var(--accent);
    border-radius: 20px;
    padding: 2px 12px;
    font-size: .78rem;
    font-weight: 600;
    margin-right: 6px;
    margin-bottom: 4px;
}
.ff-result-fire {
    background: rgba(255,59,59,.12);
    border: 1px solid var(--accent2);
    border-radius: 14px;
    padding: 1.2rem 1.6rem;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--accent2);
}
.ff-result-safe {
    background: rgba(34,197,94,.12);
    border: 1px solid var(--green);
    border-radius: 14px;
    padding: 1.2rem 1.6rem;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--green);
}
.fwi-badge {
    font-family: var(--font-h);
    font-size: 2.4rem;
    font-weight: 800;
    color: var(--accent);
}
.section-label {
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .12em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: .4rem;
}
</style>
""", unsafe_allow_html=True)

# ── Imports (after page config) ───────────────────────────────────────────────
from utils.load_model import load_models
from utils.preprocess import preprocess_input
from utils.plots import (
    
    plot_fire_risk_gauge, plot_feature_importance,
)

# ── Data & model loading ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = "D:\Jupyter\jupyter\ML\Forest_fire_classification\data\Algerian_forest_fires_dataset_update.csv"
    if not os.path.exists(path):
        st.error("❌ `data/dataset.csv` not found. Add your dataset and restart.")
        st.stop()
    return pd.read_csv(path)

df = load_data()
clf_model, reg_model, scaler, feature_names = load_models()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔥 Forest Fire\n**Prediction System**")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠  Home", "🔮  Prediction"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<div class='section-label'>Dataset</div>"
        f"<div style='font-size:.9rem'>{df.shape[0]:,} rows · {df.shape[1]} cols</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='section-label' style='margin-top:.8rem'>Features</div>"
        f"<div style='font-size:.9rem'>{len(feature_names)} model inputs</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption("Phase 2 · Streamlit UI")


# ══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.markdown("<h1 style='font-size:2.4rem'>🔥 Forest Fire<br>Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>ML-powered early warning · Algeria Dataset</div><br>", unsafe_allow_html=True)

    # KPI row
    num_df = df.select_dtypes(include="number")
    fire_col = [c for c in df.columns if "fire" in c.lower() or "classes" in c.lower()]
    fire_pct = round(df[fire_col[0]].astype(str).str.lower().str.contains("fire").mean() * 100, 1) if fire_col else "N/A"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",  f"{df.shape[0]:,}")
    c2.metric("Features",       len(feature_names))
    c3.metric("Fire Samples",   f"{fire_pct}%" if isinstance(fire_pct, float) else fire_pct)
    

    st.markdown("<br>", unsafe_allow_html=True)

    # What this system does
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown("""
<div class='ff-card'>
<h3 style='margin:0 0 .6rem'>What it predicts</h3>
<span class='ff-tag'>Classification</span><span class='ff-tag'>Regression</span><br><br>
<b>🔥 Fire Occurrence</b><br>
<small style='color:#7878a0'>Binary: fire / no fire — Logistic Regression</small><br><br>
<b>📈 FWI Score</b><br>
<small style='color:#7878a0'>Fire Weather Index intensity — Lasso Regression</small>
</div>
""", unsafe_allow_html=True)
    with col_b:
        st.markdown("""
<div class='ff-card'>
<h3 style='margin:0 0 .6rem'>Input features used</h3>
""" + "".join(f"<span class='ff-tag'>{f}</span>" for f in feature_names) + """
</div>
""", unsafe_allow_html=True)

    st.markdown("### Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True, height=300)



# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮  Prediction":
    st.markdown("<h1>🔮 Fire Risk Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Enter weather & index values below</div><br>", unsafe_allow_html=True)

    # ── Input form ──────────────────────────────────────────────────────────
    # Build metadata for smarter inputs
    num_df = df.select_dtypes(include="number")
    input_data = {}

    with st.form("prediction_form"):
        st.markdown("#### 🌡️ Input Parameters")

        # Render inputs in a 3-column grid
        cols_per_row = 3
        feature_chunks = [feature_names[i:i+cols_per_row] for i in range(0, len(feature_names), cols_per_row)]

        for chunk in feature_chunks:
            row_cols = st.columns(len(chunk))
            for col, feat in zip(row_cols, chunk):
                with col:
                    # Use dataset stats for better defaults/ranges
                    if feat in num_df.columns:
                        col_min  = float(num_df[feat].min())
                        col_max  = float(num_df[feat].max())
                        col_mean = float(num_df[feat].mean())
                        help_txt = f"Range: {col_min:.1f} – {col_max:.1f}"
                    else:
                        col_mean = 0.0
                        help_txt = ""

                    input_data[feat] = st.number_input(
                        feat, value=round(col_mean, 2),
                        help=help_txt, format="%.2f"
                    )

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔥 Run Prediction", use_container_width=True)

    # ── Prediction output ────────────────────────────────────────────────────
    if submitted:
        try:
            with st.spinner("Running models…"):
                processed = preprocess_input(input_data, scaler, feature_names)

                clf_pred  = clf_model.predict(processed)[0]
                reg_pred  = float(reg_model.predict(processed)[0])

                # Probability (classification confidence)
                if hasattr(clf_model, "predict_proba"):
                    clf_proba = clf_model.predict_proba(processed)[0]
                    fire_prob = float(clf_proba[1])
                else:
                    fire_prob = 1.0 if clf_pred == 1 else 0.0

            st.markdown("---")
            st.markdown("### 📋 Results")

            res_col, gauge_col = st.columns([1.6, 1])

            with res_col:
                # Classification result
                if clf_pred == 1:
                    st.markdown("<div class='ff-result-fire'>🔥 Fire Likely — High Alert</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='ff-result-safe'>✅ No Fire — Conditions Safe</div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # FWI score with color coding
                fwi_color = "#ff3b3b" if reg_pred > 30 else ("#ffa500" if reg_pred > 15 else "#22c55e")
                severity  = "Extreme" if reg_pred > 30 else ("Moderate" if reg_pred > 15 else "Low")
                st.markdown(
                    f"<div class='ff-card'>"
                    f"<div class='section-label'>Predicted FWI (Fire Weather Index)</div>"
                    f"<div class='fwi-badge' style='color:{fwi_color}'>{reg_pred:.2f}</div>"
                    f"<br><small style='color:{fwi_color}'>Severity: <b>{severity}</b></small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Confidence
                st.metric("Classification Confidence", f"{fire_prob*100:.1f}%",
                          delta="Fire" if clf_pred == 1 else "No Fire")

            with gauge_col:
                fig_g = plot_fire_risk_gauge(fire_prob)
                st.pyplot(fig_g, use_container_width=True)

            # ── Feature importance (if available) ───────────────────────────
            st.markdown("---")
            st.markdown("#### 🔍 Feature Importance / Model Weights")

            coef = None
            if hasattr(clf_model, "coef_"):
                coef = clf_model.coef_
            elif hasattr(clf_model, "feature_importances_"):
                coef = clf_model.feature_importances_

            if coef is not None:
                fig_fi = plot_feature_importance(feature_names, coef)
                st.pyplot(fig_fi, use_container_width=True)
            else:
                st.info("Feature importance not available for this model type.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.exception(e)
