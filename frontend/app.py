"""
Eco-Switch Intent Predictor — Streamlit Frontend
Modern eco-themed UI for predicting menstrual product switching intent.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# Allow running from project root: python -m streamlit run frontend/app.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frontend.utils import get_health, post_predict

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🌿 Eco-Switch Intent Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — eco green theme, no default Streamlit look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Global background & font ─────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        background-color: #F4F9F4;
        color: #1B3B2B;
    }

    /* ── Sidebar ──────────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(160deg, #1B5E20 0%, #2E7D32 60%, #388E3C 100%);
        color: #FFFFFF;
    }
    section[data-testid="stSidebar"] * { color: #FFFFFF !important; }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stNumberInput label { color: #E8F5E9 !important; }

    /* ── Main header card ─────────────────────────────────────────────── */
    .hero-card {
        background: linear-gradient(135deg, #2E7D32 0%, #43A047 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        color: #FFFFFF;
        box-shadow: 0 4px 20px rgba(46,125,50,0.25);
    }
    .hero-card h1 { font-size: 2.2rem; margin: 0 0 0.4rem 0; font-weight: 700; }
    .hero-card p  { font-size: 1rem; opacity: 0.9; margin: 0; }

    /* ── Section headers ──────────────────────────────────────────────── */
    .section-header {
        background: #E8F5E9;
        border-left: 5px solid #2E7D32;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin: 1.5rem 0 1rem 0;
        font-weight: 600;
        font-size: 1.05rem;
        color: #1B5E20;
    }

    /* ── KPI metric cards ─────────────────────────────────────────────── */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #C8E6C9;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2E7D32;
    }
    .metric-card .metric-label {
        font-size: 0.8rem;
        color: #558B2F;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.2rem;
    }

    /* ── Result card — positive ───────────────────────────────────────── */
    .result-positive {
        background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 100%);
        border-radius: 16px;
        padding: 1.8rem 2rem;
        color: #FFFFFF;
        box-shadow: 0 6px 24px rgba(46,125,50,0.30);
        text-align: center;
    }
    .result-positive h2 { font-size: 1.8rem; margin: 0 0 0.5rem 0; }
    .result-positive p  { opacity: 0.9; margin: 0; font-size: 1rem; }

    /* ── Result card — negative ───────────────────────────────────────── */
    .result-negative {
        background: linear-gradient(135deg, #4E342E 0%, #6D4C41 100%);
        border-radius: 16px;
        padding: 1.8rem 2rem;
        color: #FFFFFF;
        box-shadow: 0 6px 24px rgba(109,76,65,0.30);
        text-align: center;
    }
    .result-negative h2 { font-size: 1.8rem; margin: 0 0 0.5rem 0; }
    .result-negative p  { opacity: 0.9; margin: 0; font-size: 1rem; }

    /* ── Primary button ───────────────────────────────────────────────── */
    div.stButton > button {
        background: linear-gradient(135deg, #2E7D32, #43A047);
        color: #FFFFFF;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1.05rem;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
        transition: all 0.2s ease;
        box-shadow: 0 3px 10px rgba(46,125,50,0.3);
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #1B5E20, #2E7D32);
        box-shadow: 0 5px 16px rgba(46,125,50,0.4);
        transform: translateY(-1px);
    }

    /* ── Input widgets ────────────────────────────────────────────────── */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 1.5px solid #A5D6A7;
    }
    .stSelectbox > div > div:focus-within,
    .stNumberInput > div > div > input:focus {
        border-color: #2E7D32 !important;
        box-shadow: 0 0 0 2px rgba(46,125,50,0.15);
    }

    /* ── Divider ──────────────────────────────────────────────────────── */
    hr { border-color: #C8E6C9; }

    /* ── Expander ─────────────────────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: #E8F5E9 !important;
        border-radius: 8px;
        color: #1B5E20 !important;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — info & backend status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌿 Eco-Switch Predictor")
    st.markdown("---")

    health = get_health()
    if health and health.get("model_loaded"):
        st.success("✅ Backend Online")
    elif health:
        st.error("⚠️ Backend: Model not loaded")
    else:
        st.warning("🔌 Backend Offline")

    st.markdown("---")
    st.markdown("### 📊 About the Dataset")
    st.markdown(
        "- **333** real-world survey respondents\n"
        "- **~68%** intend to switch\n"
        "- **24** survey questions\n"
        "- 5-fold stratified CV\n"
        "- ROC-AUC: **0.58–0.65**"
    )
    st.markdown("---")
    st.markdown("### ℹ️ About the Model")
    st.markdown(
        "This model uses a scikit-learn pipeline with:\n"
        "- **Logistic Regression**, **Random Forest**, **XGBoost**\n"
        "- SHAP explainability\n\n"
        "> _Modest AUC is expected for a 333-row survey. "
        "The value lies in the engineering process._"
    )
    st.markdown("---")
    with st.expander("🔍 Feature Groups"):
        st.markdown(
            "**Numeric:** Age, Income, Products/cycle, Years using\n\n"
            "**Ordinal:** Flow intensity, Change frequency, Duration, Cost, Manufacturer attitude\n\n"
            "**Nominal:** Occupation, Area, Product type, Disposal method, etc.\n\n"
            "**Binary flags:** Reason given, Product reason given"
        )

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-card">
        <h1>🌿 Eco-Switch Intent Predictor</h1>
        <p>
            Predict whether a menstrual product user is likely to switch to an
            eco-friendly alternative — powered by real-world survey data from 333 respondents
            and explainable AI (SHAP).
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        '<div class="metric-card"><div class="metric-value">333</div>'
        '<div class="metric-label">Survey Respondents</div></div>',
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        '<div class="metric-card"><div class="metric-value">68%</div>'
        '<div class="metric-label">Intent to Switch</div></div>',
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        '<div class="metric-card"><div class="metric-value">22</div>'
        '<div class="metric-label">Input Features</div></div>',
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        '<div class="metric-card"><div class="metric-value">XGB</div>'
        '<div class="metric-label">Best Model</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
st.markdown('<div class="section-header">👤 Section 1 — Demographics</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    age = st.number_input("Age", min_value=17, max_value=120, value=21, step=1)
with col2:
    income = st.number_input("Monthly Income (₹)", min_value=0, value=10000, step=500)
with col3:
    occupation = st.selectbox(
        "Occupation",
        ["Student", "Service", "Business", "Homemaker", "Others"],
    )
with col4:
    area = st.selectbox("Area", ["Urban", "Rural"])

st.markdown('<div class="section-header">🛒 Section 2 — Product Usage</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    menstrual_product = st.selectbox(
        "Menstrual Product Used",
        ["Sanitary napkins", "Organic pads", "Menstrual cups", "Tampons", "Cloth"],
    )
    disposable_reusable = st.selectbox(
        "Disposable / Reusable?",
        ["Disposable", "Reusable", "Both"],
    )
with col2:
    products_per_cycle = st.number_input(
        "Products Used per Cycle", min_value=1, max_value=50, value=10, step=1
    )
    years_using = st.number_input(
        "Years Using Current Product", min_value=0, max_value=40, value=5, step=1
    )
with col3:
    duration = st.selectbox(
        "Single Product Duration per Day",
        ["1-3 hours", "4-6 hours", "More than 6 hours"],
    )
    monthly_cost = st.selectbox(
        "Monthly Cost Spend",
        ["Less than 100 Rs", "100-150 Rs", "150-200 Rs", "More than 200 Rs"],
    )

st.markdown('<div class="section-header">♻️ Section 3 — Disposal Habits</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    method_disposing = st.selectbox(
        "Method of Disposing",
        [
            "General household waste",
            "Separated dustbins for sanitary napkins",
            "Burns it",
            "Other",
        ],
    )
with col2:
    availability_machines = st.selectbox(
        "Disposal Machines Available?", ["Yes", "No"]
    )
with col3:
    reason_given = int(
        st.selectbox(
            "Do you have a reason for wanting to switch to eco-friendly products?",
            ["No", "Yes"],
        ) == "Yes"
    )
    product_reason_given = int(
        st.selectbox(
            "Do you have a specific reason for using your current product?",
            ["No", "Yes"],
        ) == "Yes"
    )

st.markdown('<div class="section-header">🌍 Section 4 — Attitudes & Awareness</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    intensity = st.selectbox("Intensity of Flow", ["Low", "Medium", "High"])
    awareness = st.selectbox("Aware of Menstrual Waste Impact?", ["Yes", "No"])
with col2:
    change_freq = st.selectbox(
        "Daily Change Frequency",
        ["Less than 3 times", "3-5 times", "More than 5 times"],
    )
    promoting_gov = st.selectbox("Promoting Gov Policies?", ["Yes", "No"])
with col3:
    any_discomfort = st.selectbox("Any Discomfort?", ["Yes", "No"])
    aware_gov = st.selectbox("Aware of Gov Policies?", ["Yes", "No"])
with col4:
    shift_to_other = st.selectbox("Willing to Shift Products?", ["Yes", "No"])
    manufacturer_eco = st.selectbox(
        "Should Manufacturers Go Eco?",
        ["Disagree", "Neutral", "Agree", "Strongly agree"],
    )

# ---------------------------------------------------------------------------
# Predict button
# ---------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🌿 Predict Switching Intent", use_container_width=True)

# ---------------------------------------------------------------------------
# Prediction result
# ---------------------------------------------------------------------------
if predict_btn:
    payload = {
        "Age": float(age),
        "Income": float(income),
        "No.of products used for a cycle": float(products_per_cycle),
        "years_using_product": float(years_using),
        "Intensity of flow": intensity,
        "Single day changing frequency": change_freq,
        "Duration of single product per day": duration,
        "Monthly cost spend": monthly_cost,
        "Manufacturer should produce eco friendly products": manufacturer_eco,
        "Occupation": occupation,
        "Area": area,
        "Menstrual product": menstrual_product,
        "Disposable/Reusable product": disposable_reusable,
        "Method of disposing": method_disposing,
        "Availability of disposable machines": availability_machines,
        "Awareness about impact of menstrual waste": awareness,
        "Promoting Gov policies": promoting_gov,
        "Any discomfort": any_discomfort,
        "Aware about Gov policies": aware_gov,
        "Shift to other product": shift_to_other,
        "reason_given": reason_given,
        "product_reason_given": product_reason_given,
    }

    with st.spinner("🌿 Analyzing your profile..."):
        result = post_predict(payload)

    st.markdown("---")

    if not result["ok"]:
        st.error(f"❌ {result['error']}")
    else:
        data = result["data"]
        intent = data["switch_intent"]
        prob = data["probability"]
        label = data["label"]
        shap_values = data.get("shap_values")

        # ── Result card ──────────────────────────────────────────────
        card_class = "result-positive" if intent == 1 else "result-negative"
        interpretation = (
            "This profile shows strong intent to switch to eco-friendly products. "
            "Key contributing factors are highlighted below."
            if intent == 1
            else "This profile is not yet ready to switch. "
            "Raising awareness and reducing cost barriers could help."
        )
        st.markdown(
            f"""
            <div class="{card_class}">
                <h2>{label}</h2>
                <p>{interpretation}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Probability gauge ────────────────────────────────────────
        prob_col, space_col = st.columns([2, 1])
        with prob_col:
            st.markdown(f"**Switching Probability: {prob * 100:.1f}%**")
            st.progress(prob)
            if prob >= 0.7:
                st.success("High likelihood of switching 🌿")
            elif prob >= 0.4:
                st.warning("Moderate likelihood — some barriers present 🌱")
            else:
                st.info("Low likelihood — awareness and cost are key barriers 🍃")

        # ── SHAP chart ───────────────────────────────────────────────
        if shap_values:
            st.markdown("---")
            st.markdown("### 🔍 Feature Contributions to Your Prediction")

            # Sort by absolute value, take top 10
            sorted_shap = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            features = [item[0] for item in sorted_shap]
            values = [item[1] for item in sorted_shap]

            # Shorten feature names for readability
            def shorten(name: str) -> str:
                replacements = {
                    "num__": "", "ord__": "", "nom__": "",
                    "bin__": "", "__": " ",
                    "Manufacturer should produce eco friendly products": "Manufacturer eco attitude",
                    "Awareness about impact of menstrual waste": "Waste awareness",
                    "Single day changing frequency": "Change frequency",
                    "Duration of single product per day": "Product duration",
                    "No.of products used for a cycle": "Products/cycle",
                    "years_using_product": "Years using",
                    "Disposable/Reusable product": "Disposable/Reusable",
                    "Availability of disposable machines": "Disposal machines",
                }
                for old, new in replacements.items():
                    name = name.replace(old, new)
                return name[:40]

            short_features = [shorten(f) for f in features]
            colors = ["#2E7D32" if v > 0 else "#795548" for v in values]

            fig, ax = plt.subplots(figsize=(9, 5))
            fig.patch.set_facecolor("#F4F9F4")
            ax.set_facecolor("#F4F9F4")

            bars = ax.barh(short_features[::-1], values[::-1], color=colors[::-1],
                           edgecolor="white", height=0.6)

            # Value labels
            for bar, val in zip(bars, values[::-1]):
                ax.text(
                    val + (0.002 if val >= 0 else -0.002),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}",
                    va="center",
                    ha="left" if val >= 0 else "right",
                    fontsize=8,
                    color="#1B3B2B",
                )

            ax.axvline(0, color="#A5D6A7", linewidth=1.5, linestyle="--")
            ax.set_xlabel("SHAP Contribution", color="#1B3B2B")
            ax.set_title("Feature Contributions to Your Prediction 🌿",
                         color="#1B5E20", fontsize=12, fontweight="bold", pad=12)
            ax.tick_params(colors="#1B3B2B")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#C8E6C9")
            ax.spines["bottom"].set_color("#C8E6C9")

            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="#2E7D32", label="Pushes toward switching ↑"),
                Patch(facecolor="#795548", label="Pushes against switching ↓"),
            ]
            ax.legend(handles=legend_elements, loc="lower right",
                      framealpha=0.8, fontsize=8)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        elif shap_values is None and intent is not None:
            st.info("ℹ️ Feature explanations are unavailable for this prediction.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<center style='color:#81C784; font-size:0.85rem;'>"
    "🌿 Eco-Switch Intent Predictor • Built with FastAPI + Streamlit • "
    "Real-world survey data from 333 respondents"
    "</center>",
    unsafe_allow_html=True,
)
