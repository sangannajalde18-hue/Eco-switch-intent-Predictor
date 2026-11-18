"""
Eco-Switch Intent Predictor — Streamlit Frontend
Professional multi-page eco-themed UI.

Pages:
  🌿 Predict       — Live prediction with SHAP explanations
  📊 Model Insights — ROC-AUC, Confusion Matrix, feature importance
  ℹ️  About          — Dataset context and project info
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frontend.utils import get_health, post_predict

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (MUST be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌿 Eco-Switch Intent Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ────────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background-color: #F4F9F4;
    color: #1B3B2B;
}
/* ── Sidebar ───────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(165deg, #1B5E20 0%, #2E7D32 55%, #388E3C 100%);
}
section[data-testid="stSidebar"] * { color: #FFFFFF !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.25) !important; }
/* ── Hero card ──────────────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 50%, #43A047 100%);
    border-radius: 18px;
    padding: 2.2rem 2.8rem;
    margin-bottom: 1.8rem;
    color: #FFFFFF;
    box-shadow: 0 6px 28px rgba(46,125,50,0.28);
}
.hero h1 { font-size: 2.4rem; margin: 0 0 0.5rem; font-weight: 800; letter-spacing: -0.5px; }
.hero p  { font-size: 1.05rem; opacity: 0.88; margin: 0; line-height: 1.6; }
/* ── KPI cards ──────────────────────────────────────────────────────────── */
.kpi {
    background: #FFFFFF;
    border: 1.5px solid #C8E6C9;
    border-radius: 14px;
    padding: 1.3rem 1.6rem;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.kpi:hover { transform: translateY(-2px); box-shadow: 0 6px 18px rgba(46,125,50,0.15); }
.kpi .val  { font-size: 2.1rem; font-weight: 800; color: #2E7D32; }
.kpi .lbl  { font-size: 0.75rem; color: #558B2F; text-transform: uppercase;
              letter-spacing: 0.08em; margin-top: 0.25rem; }
/* ── Section header ─────────────────────────────────────────────────────── */
.sec {
    background: #E8F5E9;
    border-left: 5px solid #2E7D32;
    border-radius: 8px;
    padding: 0.65rem 1.1rem;
    margin: 1.6rem 0 1rem;
    font-weight: 700;
    font-size: 1.0rem;
    color: #1B5E20;
}
/* ── Result cards ───────────────────────────────────────────────────────── */
.res-yes {
    background: linear-gradient(135deg, #1B5E20, #2E7D32, #43A047);
    border-radius: 18px; padding: 2rem 2.4rem;
    color: #FFFFFF; text-align: center;
    box-shadow: 0 8px 28px rgba(46,125,50,0.35);
}
.res-no {
    background: linear-gradient(135deg, #4E342E, #6D4C41, #795548);
    border-radius: 18px; padding: 2rem 2.4rem;
    color: #FFFFFF; text-align: center;
    box-shadow: 0 8px 28px rgba(109,76,65,0.30);
}
.res-yes h2, .res-no h2 { font-size: 1.9rem; margin: 0 0 0.6rem; font-weight: 800; }
.res-yes p,  .res-no p  { opacity: 0.9; margin: 0; font-size: 1rem; line-height: 1.55; }
/* ── Button ─────────────────────────────────────────────────────────────── */
div.stButton > button {
    background: linear-gradient(135deg, #2E7D32, #43A047);
    color: #FFF; border: none; border-radius: 11px;
    padding: 0.8rem 2.2rem; font-size: 1.08rem; font-weight: 700;
    width: 100%; cursor: pointer; transition: all 0.2s;
    box-shadow: 0 4px 14px rgba(46,125,50,0.32);
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #1B5E20, #2E7D32);
    box-shadow: 0 6px 20px rgba(46,125,50,0.42);
    transform: translateY(-1px);
}
/* ── Tab strip ──────────────────────────────────────────────────────────── */
div[data-testid="stTabs"] button {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}
/* ── Inputs ─────────────────────────────────────────────────────────────── */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    border-radius: 8px !important;
    border: 1.5px solid #A5D6A7 !important;
}
/* ── Metric box ─────────────────────────────────────────────────────────── */
.metric-box {
    background: #FFFFFF;
    border: 1.5px solid #C8E6C9;
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    text-align: center;
}
.metric-box .m-val { font-size: 1.8rem; font-weight: 800; color: #2E7D32; }
.metric-box .m-lbl { font-size: 0.78rem; color: #558B2F; text-transform: uppercase;
                      letter-spacing: 0.07em; margin-top: 0.2rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 Eco-Switch Predictor")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🌿 Predict", "📊 Model Insights", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Backend status
    health = get_health()
    if health and health.get("model_loaded"):
        st.success("✅ Backend Online")
    elif health:
        st.error("⚠️ Model not loaded")
    else:
        st.warning("🔌 Backend Offline")

    st.markdown("---")
    st.markdown("### 📋 Quick Stats")
    st.markdown("- **333** survey respondents\n- **~68%** intent to switch\n- **24** survey features\n- Best model: **Decision Tree**")
    st.markdown("---")
    st.caption("Built with FastAPI + Streamlit")


# ─────────────────────────────────────────────────────────────────────────────
# Helper — short feature name
# ─────────────────────────────────────────────────────────────────────────────
def shorten(name: str) -> str:
    for old, new in {
        "num__": "", "ord__": "", "nom__": "", "bin__": "", "__": " → ",
        "Manufacturer should produce eco friendly products": "Manufacturer eco attitude",
        "Awareness about impact of menstrual waste": "Waste awareness",
        "Single day changing frequency": "Change frequency",
        "Duration of single product per day": "Product duration/day",
        "No.of products used for a cycle": "Products per cycle",
        "years_using_product": "Years using product",
        "Disposable/Reusable product": "Disposable/Reusable",
        "Availability of disposable machines": "Disposal machines avail.",
    }.items():
        name = name.replace(old, new)
    return name[:45]


# ─────────────────────────────────────────────────────────────────────────────
# ── PAGE 1: PREDICT ──────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
if page == "🌿 Predict":

    st.markdown("""
    <div class="hero">
        <h1>🌿 Eco-Switch Intent Predictor</h1>
        <p>Fill in the survey below to find out whether a respondent is likely to switch
        to an eco-friendly menstrual product — powered by real-world data from
        <strong>333 respondents</strong> and explainable AI (SHAP).</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    for col, val, lbl in zip(
        [k1, k2, k3, k4],
        ["333", "68%", "22", "DT"],
        ["Respondents", "Switch Intent", "Features", "Best Model"],
    ):
        col.markdown(
            f'<div class="kpi"><div class="val">{val}</div>'
            f'<div class="lbl">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 1: Demographics ──────────────────────────────────────────────
    st.markdown('<div class="sec">👤 Demographics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        age = st.number_input("Age", min_value=17, max_value=70, value=21, step=1)
    with c2:
        income = st.number_input("Monthly Income (₹)", min_value=0, value=10000, step=500)
    with c3:
        occupation = st.selectbox("Occupation",
            ["Student", "Service", "Business", "Homemaker", "Others"])
    with c4:
        area = st.selectbox("Area", ["Urban", "Rural"])

    # ── Section 2: Product Usage ─────────────────────────────────────────────
    st.markdown('<div class="sec">🛒 Product Usage</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        menstrual_product = st.selectbox("Menstrual Product",
            ["Sanitary napkins", "Organic pads", "Menstrual cups", "Tampons", "Cloth"])
        disposable_reusable = st.selectbox("Disposable / Reusable?",
            ["Disposable", "Reusable", "Both"])
    with c2:
        products_per_cycle = st.number_input("Products per Cycle",
            min_value=1, max_value=50, value=10, step=1)
        years_using = st.number_input("Years Using Current Product",
            min_value=0, max_value=40, value=5, step=1)
    with c3:
        duration = st.selectbox("Single Product Duration / Day",
            ["1-3 hours", "4-6 hours", "More than 6 hours"])
        monthly_cost = st.selectbox("Monthly Cost Spend",
            ["Less than 100 Rs", "100-150 Rs", "150-200 Rs", "More than 200 Rs"])

    # ── Section 3: Disposal Habits ───────────────────────────────────────────
    st.markdown('<div class="sec">♻️ Disposal Habits</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        method_disposing = st.selectbox("Method of Disposing",
            ["General household waste",
             "Separated dustbins for sanitary napkins",
             "Burns it", "Other"])
    with c2:
        availability_machines = st.selectbox("Disposal Machines Available?", ["Yes", "No"])
    with c3:
        reason_given = int(
            st.selectbox(
                "Do you have a reason to switch to eco-friendly products?",
                ["No", "Yes"],
            ) == "Yes"
        )
        product_reason_given = int(
            st.selectbox(
                "Do you have a specific reason for your current product?",
                ["No", "Yes"],
            ) == "Yes"
        )

    # ── Section 4: Attitudes & Awareness ────────────────────────────────────
    st.markdown('<div class="sec">🌍 Attitudes & Awareness</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        intensity    = st.selectbox("Intensity of Flow", ["Low", "Medium", "High"])
        awareness    = st.selectbox("Aware of Menstrual Waste Impact?", ["Yes", "No"])
    with c2:
        change_freq  = st.selectbox("Daily Change Frequency",
            ["Less than 3 times", "3-5 times", "More than 5 times"])
        promoting_gov = st.selectbox("Supporting Gov Eco Policies?", ["Yes", "No"])
    with c3:
        any_discomfort = st.selectbox("Any Discomfort with Current Product?", ["Yes", "No"])
        aware_gov      = st.selectbox("Aware of Gov Policies?", ["Yes", "No"])
    with c4:
        shift_to_other  = st.selectbox("Open to Shifting Products?", ["Yes", "No"])
        manufacturer_eco = st.selectbox("Should Manufacturers Go Eco?",
            ["Disagree", "Neutral", "Agree", "Strongly agree"])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Predict button ───────────────────────────────────────────────────────
    if st.button("🌿 Predict Switching Intent", use_container_width=True):

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

        with st.spinner("🌿 Analyzing your profile…"):
            result = post_predict(payload)

        st.markdown("---")

        if not result["ok"]:
            st.error(f"❌ {result['error']}")
        else:
            data   = result["data"]
            intent = data["switch_intent"]
            prob   = data["probability"]
            label  = data["label"]
            shap_values = data.get("shap_values")

            # Result card
            if intent == 1:
                st.markdown(f"""
                <div class="res-yes">
                    <h2>{label}</h2>
                    <p>This profile shows strong intent to switch to eco-friendly products.
                    The key contributing factors are highlighted below.</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="res-no">
                    <h2>{label}</h2>
                    <p>This profile is not yet ready to switch. Raising awareness and
                    reducing cost barriers could make the difference.</p>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Probability gauge
            prob_col, _ = st.columns([2, 1])
            with prob_col:
                st.markdown(f"**Switching Probability: {prob * 100:.1f}%**")
                st.progress(prob)
                if prob >= 0.70:
                    st.success("High likelihood of switching 🌿")
                elif prob >= 0.40:
                    st.warning("Moderate likelihood — some barriers present 🌱")
                else:
                    st.info("Low likelihood — awareness and cost are key barriers 🍃")

            # SHAP chart
            if shap_values:
                st.markdown("---")
                st.markdown("### 🔍 Feature Contributions to This Prediction")

                sorted_shap = sorted(shap_values.items(),
                                     key=lambda x: abs(x[1]), reverse=True)[:10]
                feats  = [shorten(item[0]) for item in sorted_shap]
                vals   = [item[1] for item in sorted_shap]
                colors = ["#2E7D32" if v > 0 else "#795548" for v in vals]

                fig, ax = plt.subplots(figsize=(9, 5))
                fig.patch.set_facecolor("#F4F9F4")
                ax.set_facecolor("#F4F9F4")
                bars = ax.barh(feats[::-1], vals[::-1], color=colors[::-1],
                               edgecolor="white", height=0.62)
                for bar, val in zip(bars, vals[::-1]):
                    ax.text(val + (0.003 if val >= 0 else -0.003),
                            bar.get_y() + bar.get_height() / 2,
                            f"{val:+.3f}", va="center",
                            ha="left" if val >= 0 else "right",
                            fontsize=8, color="#1B3B2B")
                ax.axvline(0, color="#A5D6A7", lw=1.5, ls="--")
                ax.set_xlabel("SHAP Contribution", color="#1B3B2B")
                ax.set_title("Feature Contributions to Your Prediction 🌿",
                             color="#1B5E20", fontsize=12, fontweight="bold", pad=12)
                ax.tick_params(colors="#1B3B2B")
                for spine in ["top", "right"]:
                    ax.spines[spine].set_visible(False)
                ax.spines["left"].set_color("#C8E6C9")
                ax.spines["bottom"].set_color("#C8E6C9")
                ax.legend(handles=[
                    Patch(facecolor="#2E7D32", label="Pushes toward switching ↑"),
                    Patch(facecolor="#795548", label="Pushes against switching ↓"),
                ], loc="lower right", framealpha=0.85, fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            elif intent is not None:
                st.info("ℹ️ Feature explanations unavailable for this prediction.")


# ─────────────────────────────────────────────────────────────────────────────
# ── PAGE 2: MODEL INSIGHTS ───────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Model Insights":

    st.markdown("""
    <div class="hero">
        <h1>📊 Model Insights</h1>
        <p>Performance metrics, ROC-AUC curves, confusion matrix, and feature
        importance for the trained models on the eco-switch survey dataset.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Reproduce metrics from the notebook ──────────────────────────────────
    @st.cache_data(show_spinner="Loading dataset and training models…")
    def load_metrics():
        import joblib
        from sklearn.metrics import (
            classification_report, confusion_matrix,
            roc_auc_score, roc_curve,
        )
        from sklearn.model_selection import train_test_split

        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "eco_menstrual_model_pipeline.pkl",
        )
        pipeline = joblib.load(model_path)

        df = pd.read_csv(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "menstrual.csv",
            ),
            encoding="cp1252",
        )

        # Same cleaning as notebook
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str).str.strip().replace({"": np.nan, "nan": np.nan})

        df["Income"] = pd.to_numeric(
            df["Income"].astype(str).str.replace(",", "", regex=False), errors="coerce"
        )
        df["No.of products used for a cycle"] = pd.to_numeric(
            df["No.of products used for a cycle"]
            .astype(str)
            .replace("More than 20", "25"),
            errors="coerce",
        )
        df["years_using_product"] = pd.to_numeric(
            df["No.of years using menstrual product"]
            .astype(str)
            .str.extract(r"(\d+)")[0],
            errors="coerce",
        )
        df = df.drop(columns=["No.of years using menstrual product"])

        df["switch_intent"] = df["Switching to eco friendly menstrual product"].map(
            {"Yes": 1, "Maybe": 0, "No": 0}
        )
        df["reason_given"] = df["Reasons"].notna().astype(int)
        df["product_reason_given"] = df["Reason for particular product"].notna().astype(int)

        drop_cols = [
            "Switching to eco friendly menstrual product",
            "Reasons",
            "Reason for particular product",
        ]
        df = df.drop(columns=drop_cols)

        # Rename awareness col to match pipeline (trailing space)
        if "Awareness about impact of menstrual waste" in df.columns:
            df = df.rename(columns={
                "Awareness about impact of menstrual waste":
                "Awareness about impact of menstrual waste "
            })

        y = df["switch_intent"]
        X = df.drop(columns=["switch_intent"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        y_pred      = pipeline.predict(X_test)
        y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True)
        cm     = confusion_matrix(y_test, y_pred)
        auc    = roc_auc_score(y_test, y_pred_prob)
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

        return report, cm, auc, fpr, tpr, pipeline

    try:
        report, cm, auc, fpr, tpr, pipeline = load_metrics()

        # ── Top metric cards ──────────────────────────────────────────────
        acc  = report["accuracy"]
        prec = report.get("1", report.get("1.0", {})).get("precision", 0)
        rec  = report.get("1", report.get("1.0", {})).get("recall", 0)
        f1   = report.get("1", report.get("1.0", {})).get("f1-score", 0)

        m1, m2, m3, m4 = st.columns(4)
        for col, val, lbl in zip(
            [m1, m2, m3, m4],
            [f"{acc:.1%}", f"{prec:.1%}", f"{rec:.1%}", f"{f1:.2f}"],
            ["Accuracy", "Precision (class 1)", "Recall (class 1)", "F1 Score (class 1)"],
        ):
            col.markdown(
                f'<div class="metric-box"><div class="m-val">{val}</div>'
                f'<div class="m-lbl">{lbl}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Tabs ──────────────────────────────────────────────────────────
        tab1, tab2, tab3 = st.tabs(
            ["📈 ROC-AUC Curve", "🔲 Confusion Matrix", "🌳 Feature Importance"]
        )

        # ── ROC-AUC ───────────────────────────────────────────────────────
        with tab1:
            st.markdown("### ROC-AUC Curve")
            st.caption(
                "The ROC curve shows the trade-off between true positive rate (sensitivity) "
                "and false positive rate across all thresholds. "
                "AUC = area under the curve (1.0 is perfect, 0.5 is random)."
            )
            fig, ax = plt.subplots(figsize=(7, 5))
            fig.patch.set_facecolor("#F4F9F4")
            ax.set_facecolor("#F4F9F4")
            ax.plot(fpr, tpr, color="#2E7D32", lw=2.5,
                    label=f"Decision Tree (AUC = {auc:.3f})")
            ax.plot([0, 1], [0, 1], color="#BCAAA4", lw=1.5, ls="--", label="Random Chance")
            ax.fill_between(fpr, tpr, alpha=0.12, color="#81C784")
            ax.set_xlabel("False Positive Rate", color="#1B3B2B", fontsize=11)
            ax.set_ylabel("True Positive Rate", color="#1B3B2B", fontsize=11)
            ax.set_title("ROC-AUC Curve — Eco-Switch Intent Model",
                         color="#1B5E20", fontsize=13, fontweight="bold", pad=14)
            ax.legend(fontsize=10, framealpha=0.85)
            ax.tick_params(colors="#1B3B2B")
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            ax.spines["left"].set_color("#C8E6C9")
            ax.spines["bottom"].set_color("#C8E6C9")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            col_a, col_b = st.columns(2)
            col_a.metric("AUC Score", f"{auc:.4f}")
            col_b.info(
                "AUC of **0.58–0.75** is realistic for a 333-row survey dataset. "
                "The value of this project lies in the engineering process and explainability."
            )

        # ── Confusion Matrix ──────────────────────────────────────────────
        with tab2:
            st.markdown("### Confusion Matrix")
            st.caption(
                "Rows = actual labels, Columns = predicted labels. "
                "Class 0 = Not Ready to Switch, Class 1 = Likely to Switch."
            )
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            fig.patch.set_facecolor("#F4F9F4")
            cmap = plt.cm.Greens
            im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
            plt.colorbar(im, ax=ax)
            classes = ["Not Ready (0)", "Likely Switch (1)"]
            tick_marks = np.arange(len(classes))
            ax.set_xticks(tick_marks)
            ax.set_xticklabels(classes, rotation=15, ha="right", color="#1B3B2B")
            ax.set_yticks(tick_marks)
            ax.set_yticklabels(classes, color="#1B3B2B")
            thresh = cm.max() / 2.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "#1B3B2B",
                            fontsize=14, fontweight="bold")
            ax.set_ylabel("Actual Label", color="#1B3B2B", fontsize=11)
            ax.set_xlabel("Predicted Label", color="#1B3B2B", fontsize=11)
            ax.set_title("Confusion Matrix", color="#1B5E20", fontsize=13,
                         fontweight="bold", pad=12)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            tn, fp, fn, tp = cm.ravel()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("True Positives",  str(tp), help="Correctly predicted 'Likely to Switch'")
            c2.metric("True Negatives",  str(tn), help="Correctly predicted 'Not Ready'")
            c3.metric("False Positives", str(fp), help="Predicted Switch but was Not Ready")
            c4.metric("False Negatives", str(fn), help="Predicted Not Ready but was Switch")

        # ── Feature Importance ────────────────────────────────────────────
        with tab3:
            st.markdown("### Feature Importance")
            st.caption(
                "Gini-based feature importance from the Decision Tree classifier. "
                "Higher = more influence on the prediction."
            )
            try:
                clf = pipeline.named_steps.get("clf", pipeline[-1])
                preprocessor = pipeline[:-1]
                feature_names = list(preprocessor.get_feature_names_out())
                importances   = clf.feature_importances_

                top_n = 15
                indices = np.argsort(importances)[::-1][:top_n]
                top_names  = [shorten(feature_names[i]) for i in indices]
                top_imps   = importances[indices]

                fig, ax = plt.subplots(figsize=(9, 6))
                fig.patch.set_facecolor("#F4F9F4")
                ax.set_facecolor("#F4F9F4")
                colors = [
                    "#2E7D32" if i == 0 else
                    "#43A047" if i < 3 else
                    "#81C784" if i < 7 else
                    "#A5D6A7"
                    for i in range(top_n)
                ]
                ax.barh(top_names[::-1], top_imps[::-1],
                        color=colors[::-1], edgecolor="white", height=0.65)
                ax.set_xlabel("Gini Importance", color="#1B3B2B", fontsize=11)
                ax.set_title(f"Top {top_n} Feature Importances — Decision Tree",
                             color="#1B5E20", fontsize=13, fontweight="bold", pad=12)
                ax.tick_params(colors="#1B3B2B")
                for spine in ["top", "right"]:
                    ax.spines[spine].set_visible(False)
                ax.spines["left"].set_color("#C8E6C9")
                ax.spines["bottom"].set_color("#C8E6C9")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            except Exception as e:
                st.warning(f"Could not extract feature importances: {e}")

    except Exception as exc:
        st.error(f"Could not load model metrics: {exc}")
        st.info("Make sure the backend is running and `menstrual.csv` is in the project root.")


# ─────────────────────────────────────────────────────────────────────────────
# ── PAGE 3: ABOUT ────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
elif page == "ℹ️ About":

    st.markdown("""
    <div class="hero">
        <h1>ℹ️ About This Project</h1>
        <p>An end-to-end machine learning application built on real-world survey data.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### 🎯 Project Goal")
        st.markdown(
            "Predict whether a menstrual product user is likely to switch to an "
            "eco-friendly alternative, using survey responses as features. "
            "This can help NGOs, health-tech startups, or sustainable-product brands "
            "identify which demographic and behavioral segments to target."
        )

        st.markdown("### 📊 Dataset")
        st.markdown(
            "- **333** real-world survey respondents\n"
            "- **24** questions covering demographics, product usage, disposal habits, "
            "and environmental attitudes\n"
            "- Target: `Yes` (226) → 1, `Maybe` (85) + `No` (22) → 0\n"
            "- Positive class rate: **67.9%**"
        )

        st.markdown("### 🤖 Models Trained")
        metrics_data = {
            "Model": ["Logistic Regression", "Random Forest", "Decision Tree (best)"],
            "Accuracy": ["63%", "67%", "64%"],
            "Notes": [
                "Interpretable baseline",
                "Best recall on class 1",
                "Saved to pipeline · used for deployment",
            ],
        }
        st.dataframe(pd.DataFrame(metrics_data), hide_index=True, use_container_width=True)

        st.markdown("### ⚙️ Tech Stack")
        st.markdown(
            "- **Backend:** FastAPI + Pydantic v2 + joblib\n"
            "- **Frontend:** Streamlit with custom CSS\n"
            "- **ML:** scikit-learn Pipeline + ColumnTransformer\n"
            "- **Explainability:** SHAP\n"
            "- **Deployment:** Render"
        )

    with col2:
        st.markdown("### 📐 Feature Groups")
        with st.expander("🔢 Numeric (4)", expanded=True):
            st.markdown("- Age\n- Income\n- Products per cycle\n- Years using product")
        with st.expander("📶 Ordinal (5)"):
            st.markdown(
                "- Intensity of flow\n- Changing frequency\n- Product duration\n"
                "- Monthly cost\n- Manufacturer eco attitude"
            )
        with st.expander("🏷️ Nominal (11)"):
            st.markdown(
                "- Occupation · Area\n- Menstrual product\n- Disposable/Reusable\n"
                "- Method of disposing\n- Disposal machines\n- Waste awareness\n"
                "- Gov policy support · Discomfort\n- Gov policy awareness\n"
                "- Shift willingness"
            )
        with st.expander("🔘 Binary flags (2)"):
            st.markdown(
                "- `reason_given` — has a reason to switch\n"
                "- `product_reason_given` — has a reason for current product"
            )

        st.markdown("### ⚠️ Honest Model Note")
        st.info(
            "ROC-AUC of 0.58–0.75 is realistic for a 333-row survey dataset. "
            "The value of this project lies in the **engineering process** and "
            "**SHAP explainability**, not a claim of high predictive accuracy."
        )

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#81C784; font-size:0.82rem;'>"
    "🌿 Eco-Switch Intent Predictor &nbsp;•&nbsp; FastAPI + Streamlit &nbsp;•&nbsp; "
    "333 real-world survey respondents</center>",
    unsafe_allow_html=True,
)
