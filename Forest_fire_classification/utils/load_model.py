import joblib
import streamlit as st
import os

@st.cache_resource
def load_models():
    """Load all model artifacts with caching and error handling."""
    required_files = {
        "clf_model": "models/clf_model.pkl",
        "reg_model": "models/reg_model.pkl",
        "scaler":    "models/scaler.pkl",
        "features":  "models/features.pkl",
    }

    missing = [path for path in required_files.values() if not os.path.exists(path)]
    if missing:
        st.error(f"❌ Missing model files: {missing}\n\nRun Phase 1 training first to generate `.pkl` files.")
        st.stop()

    clf_model = joblib.load(required_files["clf_model"])
    reg_model = joblib.load(required_files["reg_model"])
    scaler    = joblib.load(required_files["scaler"])
    features  = joblib.load(required_files["features"])

    return clf_model, reg_model, scaler, features
