"""
TruthLenseAI — Fake News Detector
app.py  |  Streamlit UI (Dual Light/Dark Theme & Native Tabs)
"""

import os
import json
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import requests
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TruthLenseAI",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# Theme State Management
# ─────────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

def toggle_theme():
    if st.session_state.theme == "Dark":
        st.session_state.theme = "Light"
    else:
        st.session_state.theme = "Dark"

# ─────────────────────────────────────────────────────────────
# Theme Palettes
# ─────────────────────────────────────────────────────────────
if st.session_state.theme == "Dark":
    bg_main = "#09090B"         # Very dark, almost black
    bg_card = "#18181B"         # Slightly lighter dark for cards
    text_main = "#FAFAFA"       # Off-white text
    text_muted = "#A1A1AA"      # Muted gray text
    primary = "#8B5CF6"         # Vibrant Purple (Accent)
    primary_hover = "#7C3AED"
    border = "#27272A"          # Subtle border
    success_bg = "rgba(16, 185, 129, 0.1)"
    success_border = "rgba(16, 185, 129, 0.2)"
    success_text = "#34D399"
    danger_bg = "rgba(239, 68, 68, 0.1)"
    danger_border = "rgba(239, 68, 68, 0.2)"
    danger_text = "#F87171"
else:
    bg_main = "#FAFAFA"         # Clean off-white
    bg_card = "#FFFFFF"         # Pure white cards
    text_main = "#09090B"       # Very dark text
    text_muted = "#71717A"      # Muted gray text
    primary = "#3B82F6"         # Vibrant Blue (Accent)
    primary_hover = "#2563EB"
    border = "#E4E4E7"          # Subtle border
    success_bg = "rgba(16, 185, 129, 0.05)"
    success_border = "rgba(16, 185, 129, 0.15)"
    success_text = "#059669"
    danger_bg = "rgba(239, 68, 68, 0.05)"
    danger_border = "rgba(239, 68, 68, 0.15)"
    danger_text = "#DC2626"

# ─────────────────────────────────────────────────────────────
# Dynamic CSS Injection
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700&display=swap');

/* Global Styles */
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif !important;
}}

/* Background and Text Colors */
.stApp, [data-testid="stAppViewContainer"] {{
    background-color: {bg_main} !important;
    color: {text_main} !important;
}}

/* Hide Sidebar entirely */
[data-testid="stSidebar"] {{
    display: none !important;
}}
[data-testid="collapsedControl"] {{
    display: none !important;
}}

/* Headers */
h1, h2, h3, h4, h5, h6 {{
    font-family: 'Outfit', sans-serif !important;
    color: {text_main} !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em;
}}

/* Modern Pill-Shaped Navigation Tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    background-color: {bg_card} !important;
    border-radius: 100px !important;
    padding: 6px 10px !important;
    border: 1px solid {border} !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04) !important;
    gap: 8px !important;
    margin-bottom: 25px !important;
    display: inline-flex !important;
}}

/* Hide default bottom red/blue line */
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
    display: none !important;
}}

/* Individual Tab Buttons */
[data-testid="stTabs"] [data-baseweb="tab"] {{
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.05rem !important;
    color: {text_muted} !important;
    font-weight: 500 !important;
    padding: 10px 24px !important;
    border-radius: 100px !important;
    background-color: transparent !important;
    border: none !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    margin: 0 !important;
}}

/* Active Tab */
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {{
    background-color: {primary} !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 12px {primary}55 !important;
}}

/* Hover for inactive tabs */
[data-testid="stTabs"] [data-baseweb="tab"]:not([aria-selected="true"]):hover {{
    color: {text_main} !important;
    background-color: {border} !important;
}}

/* Text Areas and Inputs */
.stTextArea textarea {{
    background-color: {bg_card} !important;
    color: {text_main} !important;
    border: 1px solid {border} !important;
    border-radius: 10px !important;
    padding: 15px !important;
    font-size: 1.05rem !important;
    -webkit-text-fill-color: {text_main} !important;
}}
.stTextArea textarea:focus {{
    border-color: {primary} !important;
    box-shadow: 0 0 0 2px {primary}33 !important;
}}

/* Buttons */
.stButton > button {{
    background-color: {primary} !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    font-family: 'Outfit', sans-serif !important;
    transition: all 0.2s ease !important;
}}
.stButton > button:hover {{
    background-color: {primary_hover} !important;
    transform: translateY(-2px) !important;
}}

/* Theme Toggle Button specific styling */
.theme-toggle-btn {{
    display: flex;
    justify-content: flex-end;
    margin-right: -10px;
    margin-top: 10px;
}}
.theme-toggle-btn .stButton > button {{
    background-color: transparent !important;
    color: {text_main} !important;
    border: 1px solid {border} !important;
    box-shadow: none !important;
    border-radius: 50% !important;
    width: 36px !important;
    height: 36px !important;
    min-width: 36px !important;
    min-height: 36px !important;
    max-width: 36px !important;
    max-height: 36px !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 1.1rem !important;
}}
.theme-toggle-btn .stButton > button p {{
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1 !important;
}}
.theme-toggle-btn .stButton > button:hover {{
    background-color: {bg_card} !important;
    border-color: {primary} !important;
}}

/* Cards / Containers */
.card-container {{
    background-color: {bg_card};
    border: 1px solid {border};
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    transition: transform 0.2s, border-color 0.2s;
}}
.card-container:hover {{
    transform: translateY(-2px);
    border-color: {primary};
}}

/* Custom Alert Boxes */
.alert-box {{
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    font-weight: 500;
}}
.alert-real {{
    background: {success_bg};
    border: 1px solid {success_border};
    color: {success_text};
}}
.alert-fake {{
    background: {danger_bg};
    border: 1px solid {danger_border};
    color: {danger_text};
}}

/* Metrics */
.metric-value {{
    font-size: 2.5rem;
    font-weight: 700;
    color: {primary};
    font-family: 'Outfit', sans-serif;
}}
.metric-label {{
    font-size: 0.9rem;
    color: {text_muted};
    text-transform: uppercase;
    letter-spacing: 1px;
}}

/* DataFrame */
[data-testid="stDataFrame"] {{
    background-color: {bg_card} !important;
    border-radius: 10px;
    border: 1px solid {border};
}}
[data-testid="stTable"] {{
    color: {text_main} !important;
}}

/* Header padding */
#root > div:first-child {{
    padding-top: 2rem !important;
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Constants & Data Loading
# ─────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join("models", "fake_news_model.joblib")
METRICS_PATH = os.path.join("reports", "metrics.json")
CM_PATH      = os.path.join("reports", "confusion_matrix.png")
DATA_DIR     = "Data"

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data(show_spinner=False)
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_dataset():
    csvs = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csvs:
        return None
    return pd.read_csv(csvs[0])

model   = load_model()
metrics = load_metrics()
df_raw  = load_dataset()

# ─────────────────────────────────────────────────────────────
# Top Bar: Title + Theme Toggle
# ─────────────────────────────────────────────────────────────
head_col1, head_col2 = st.columns([10, 1])
with head_col1:
    st.title("TruthLenseAI")
    st.markdown(f"<p style='color: {text_muted}; font-size: 1.1rem; margin-top: -10px; margin-bottom: 20px;'>Advanced Fake News Detection powered by NLP</p>", unsafe_allow_html=True)
with head_col2:
    st.markdown('<div class="theme-toggle-btn">', unsafe_allow_html=True)
    theme_icon = "🌞" if st.session_state.theme == "Dark" else "🌙"
    if st.button(theme_icon, key="theme_toggle"):
        toggle_theme()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Main Layout using Native Tabs
# ─────────────────────────────────────────────────────────────
tab_home, tab_detect, tab_insights, tab_about = st.tabs(["🏠 Home", "🔍 Detect News", "📊 Model Insights", "ℹ️ About"])

# ═══════════════════════════════════════════════════════════════
# TAB: Home
# ═══════════════════════════════════════════════════════════════
with tab_home:
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.2, 1], gap="large")
    
    with col1:
        st.markdown(f"""
        <h1 style='font-size: 3.8rem; line-height: 1.15; margin-bottom: 1.2rem; font-weight: 700;'>
            Identify the truth in <br>
            <span style='color: {primary};'>modern media.</span>
        </h1>
        <p style='font-size: 1.15rem; color: {text_muted}; line-height: 1.6; margin-bottom: 2rem;'>
            TruthLenseAI utilizes advanced Natural Language Processing to analyze news headlines, 
            detecting linguistic patterns associated with sensationalism, clickbait, and fake news.
            Experience transparent, machine-learning driven classification.
        </p>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="card-container">
            <h3 style="margin-top: 0;">How it works</h3>
            <div style="margin-top: 1.5rem;">
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <div style="background: {border}; color: {text_main}; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; margin-right: 15px;">1</div>
                    <div style="color: {text_muted}; font-weight: 500;">Input any news headline</div>
                </div>
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <div style="background: {border}; color: {text_main}; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; margin-right: 15px;">2</div>
                    <div style="color: {text_muted}; font-weight: 500;">NLP Feature Extraction (TF-IDF)</div>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="background: {primary}; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; margin-right: 15px; color: white;">3</div>
                    <div style="color: {text_muted}; font-weight: 500;">Instant classification with confidence score</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown(f"<br><hr style='border-color: {border};'><br>", unsafe_allow_html=True)
    
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(f"""
        <div class="card-container">
            <h3 style="color: {primary}; margin-top:0;">⚡ Lightning Fast</h3>
            <p style="color: {text_muted};">Optimized scikit-learn pipeline delivers predictions in milliseconds without relying on external APIs.</p>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown(f"""
        <div class="card-container">
            <h3 style="color: {primary}; margin-top:0;">🔍 Transparent</h3>
            <p style="color: {text_muted};">See exactly how confident the model is with probability distributions for every prediction.</p>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown(f"""
        <div class="card-container">
            <h3 style="color: {primary}; margin-top:0;">📊 Data Driven</h3>
            <p style="color: {text_muted};">Trained on over 23,000 verified examples from the benchmark FakeNewsNet dataset.</p>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB: Detect News
# ═══════════════════════════════════════════════════════════════
with tab_detect:
    st.markdown("<h3>Analyze Content</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {text_muted};'>Provide news content below to determine its authenticity.</p>", unsafe_allow_html=True)
    
    if model is None:
        st.error("Model not found. Please run the training script first.")
    else:
        input_type = st.radio("Choose Input Type:", ["Headline", "Full Article", "URL Link"], horizontal=True)
        
        user_input = ""
        if input_type == "Headline":
            user_input = st.text_area("Headline", placeholder="e.g. Breaking: Major celebrity announces secret project...", label_visibility="collapsed", height=120)
        elif input_type == "Full Article":
            user_input = st.text_area("Full Article Text", placeholder="Paste the full article content here...", label_visibility="collapsed", height=250)
        elif input_type == "URL Link":
            user_input = st.text_input("Article URL", placeholder="https://www.example.com/news-article", label_visibility="collapsed")
        
        c1, c2, c3 = st.columns([1, 1, 4])
        with c1:
            analyze_btn = st.button("Analyze Content", use_container_width=True)
            
        if analyze_btn:
            if not user_input.strip():
                st.warning("Please provide input to analyze.")
            else:
                with st.spinner("Processing..."):
                    cleaned = user_input.strip()
                    
                    if input_type == "URL Link":
                        try:
                            st.info(f"Extracting text from: {cleaned}")
                            headers = {'User-Agent': 'Mozilla/5.0'}
                            response = requests.get(cleaned, headers=headers, timeout=10)
                            response.raise_for_status()
                            soup = BeautifulSoup(response.text, 'html.parser')
                            paragraphs = soup.find_all('p')
                            extracted_text = ' '.join([p.get_text() for p in paragraphs]).strip()
                            if not extracted_text:
                                st.warning("Could not extract paragraph text from the provided URL. The page might be protected or not contain standard article paragraphs.")
                                st.stop()
                            cleaned = extracted_text
                        except Exception as e:
                            st.error(f"Error fetching URL: {e}")
                            st.stop()
                            
                    prediction = model.predict([cleaned])[0]
                    has_proba = hasattr(model, "predict_proba")
                    
                    if has_proba:
                        proba = model.predict_proba([cleaned])[0]
                        fake_prob = proba[0] * 100
                        real_prob = proba[1] * 100
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    if prediction == 1:
                        st.markdown(f"""
                        <div class="alert-box alert-real">
                            <h2 style="color: {success_text}; margin-top: 0;">✅ Likely Real</h2>
                            <p style="margin-bottom: 0;">The model predicts this headline matches patterns of credible news.</p>
                            {f"<div style='margin-top: 15px; font-weight: 600;'>Confidence: {real_prob:.1f}%</div>" if has_proba else ""}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="alert-box alert-fake">
                            <h2 style="color: {danger_text}; margin-top: 0;">🚨 Likely Fake</h2>
                            <p style="margin-bottom: 0;">The model predicts this headline matches patterns of tabloid or exaggerated news.</p>
                            {f"<div style='margin-top: 15px; font-weight: 600;'>Confidence: {fake_prob:.1f}%</div>" if has_proba else ""}
                        </div>
                        """, unsafe_allow_html=True)
                        
                if has_proba:
                    st.markdown("<br><h4>Probability Distribution</h4>", unsafe_allow_html=True)
                    st.progress(int(real_prob), text="Real")
                    st.progress(int(fake_prob), text="Fake")

# ═══════════════════════════════════════════════════════════════
# TAB: Model Insights
# ═══════════════════════════════════════════════════════════════
with tab_insights:
    if metrics is None:
        st.warning("No metrics found. Please train the model to generate reports.")
    else:
        st.markdown("<h3>Performance Metrics</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: {text_muted};'>Based on held-out test data (20% split)</p>", unsafe_allow_html=True)
        
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"<div class='card-container'><div class='metric-value'>{metrics['accuracy']*100:.1f}%</div><div class='metric-label'>Accuracy</div></div>", unsafe_allow_html=True)
        with m2:
            st.markdown(f"<div class='card-container'><div class='metric-value'>{metrics['f1']*100:.1f}%</div><div class='metric-label'>F1 Score</div></div>", unsafe_allow_html=True)
        with m3:
            st.markdown(f"<div class='card-container'><div class='metric-value'>{metrics['precision']*100:.1f}%</div><div class='metric-label'>Precision</div></div>", unsafe_allow_html=True)
        with m4:
            st.markdown(f"<div class='card-container'><div class='metric-value'>{metrics['recall']*100:.1f}%</div><div class='metric-label'>Recall</div></div>", unsafe_allow_html=True)
            
        st.markdown(f"<br><hr style='border-color: {border};'><br>", unsafe_allow_html=True)
        
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.markdown("<h4>Confusion Matrix</h4>", unsafe_allow_html=True)
            if os.path.exists(CM_PATH):
                # FIXED: use_container_width instead of use_column_width to prevent Streamlit exception
                st.image(CM_PATH, use_container_width=True)
            else:
                st.info("Confusion matrix image not found.")
                
        with c2:
            st.markdown("<h4>Model Architecture</h4>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="card-container">
                <p><b>Best Model selected:</b> <span style="color:{primary};">{metrics.get('best_model', 'N/A')}</span></p>
                <p><b>Feature Extraction:</b> TF-IDF Vectorizer</p>
                <p><b>Dataset:</b> FakeNewsNet GossipCop</p>
            </div>
            """, unsafe_allow_html=True)
            
            if "class_report" in metrics:
                st.markdown("<h4>Detailed Classification Report</h4>", unsafe_allow_html=True)
                cr = metrics["class_report"]
                df_cr = pd.DataFrame(cr).transpose()
                st.dataframe(df_cr.style.format("{:.3f}"), use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# TAB: About
# ═══════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("<h3>About the Project</h3>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="card-container">
        <h4>TruthLenseAI</h4>
        <p style="color: {text_muted}; line-height: 1.7;">
        TruthLenseAI is designed to combat misinformation by providing an accessible interface for Natural Language Processing algorithms. 
        It demonstrates how machine learning models, specifically those utilizing TF-IDF and Naive Bayes/Logistic Regression, 
        can identify linguistic markers often present in fabricated news headlines.
        </p>
        <p style="color: {text_muted}; line-height: 1.7;">
        <b>Note on limitations:</b> The current model analyzes <i>headlines only</i> and does not fact-check against live databases. 
        It is trained primarily on celebrity and entertainment news (GossipCop). High confidence scores reflect linguistic similarity to training data, 
        not verified truth.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h4>Tech Stack</h4>", unsafe_allow_html=True)
    
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.markdown(f"<div class='card-container' style='text-align:center;'><b>Python</b><br><span style='color:{text_muted};font-size:0.9rem;'>Core Logic</span></div>", unsafe_allow_html=True)
    with t2:
        st.markdown(f"<div class='card-container' style='text-align:center;'><b>Streamlit</b><br><span style='color:{text_muted};font-size:0.9rem;'>Frontend UI</span></div>", unsafe_allow_html=True)
    with t3:
        st.markdown(f"<div class='card-container' style='text-align:center;'><b>Scikit-Learn</b><br><span style='color:{text_muted};font-size:0.9rem;'>ML Pipeline</span></div>", unsafe_allow_html=True)
    with t4:
        st.markdown(f"<div class='card-container' style='text-align:center;'><b>Pandas & Numpy</b><br><span style='color:{text_muted};font-size:0.9rem;'>Data Processing</span></div>", unsafe_allow_html=True)