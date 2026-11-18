# 🌿 Eco-Switch Intent Predictor

An end-to-end machine learning web application that predicts whether a menstrual product
user is likely to switch to an eco-friendly alternative — built on real-world survey data
from **333 respondents**, with a FastAPI backend, modern Streamlit frontend, and SHAP explanations.

---

## 🖼️ Application Pages

| Page | Description |
|------|-------------|
| **🌿 Predict** | Fill in a 22-field survey form and get a live prediction with probability gauge and SHAP feature chart |
| **📊 Model Insights** | ROC-AUC curve, Confusion Matrix, and Feature Importance for the trained Decision Tree model |
| **ℹ️ About** | Dataset context, model comparison table, tech stack, and feature group breakdown |

---

## 📊 Dataset & Model

- **333** real-world survey respondents
- **24** survey questions: demographics, product usage, disposal habits, environmental attitudes
- **Target:** `Yes` → 1 (intends to switch), `Maybe` / `No` → 0
- **Class distribution:** ~68% positive (intend to switch)
- **Models trained:** Logistic Regression, Random Forest, Decision Tree
- **Deployed model:** Decision Tree (max_depth=4, class_weight='balanced')
- **Honest AUC note:** 0.58–0.75 is realistic for a 333-row survey dataset

---

## 🗂️ Project Structure

```
.
├── backend/
│   ├── __init__.py
│   ├── main.py        # FastAPI app — routes, CORS, lifespan loader
│   ├── schemas.py     # Pydantic v2 request/response models (22 fields)
│   └── services.py    # Prediction logic, DataFrame builder, SHAP explainer
├── frontend/
│   ├── __init__.py
│   ├── app.py         # Streamlit multi-page UI
│   ├── utils.py       # API call helpers (get_health, post_predict)
│   └── .streamlit/
│       └── secrets.toml
├── eco_switch_intent_pipeline.joblib  ← pre-trained sklearn pipeline
├── menstrual.csv                      ← survey dataset (used for Model Insights page)
├── requirements.txt
├── render.yaml                        ← Render Blueprint for deployment
├── .python-version                    ← pins Python 3.12 on Render
├── .gitignore
└── README.md
```

---

## ⚡ Local Setup

### 1. Create a Python 3.12 virtual environment
```bash
py -3.12 -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
```

### 2. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Start the FastAPI backend (Terminal 1)
```bash
uvicorn backend.main:app --reload --port 8000
```
- Health check: http://localhost:8000/health → `{"status":"ok","model_loaded":true}`
- Swagger docs: http://localhost:8000/docs

### 4. Start the Streamlit frontend (Terminal 2)
```bash
streamlit run frontend/app.py
```
- App: http://localhost:8501

> Both terminals must stay open simultaneously.

---

## 🚀 Deploy to Render

### Backend service
| Field | Value |
|---|---|
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` |
| Env: `MODEL_PATH` | `eco_switch_intent_pipeline.joblib` |
| Env: `ALLOWED_ORIGINS` | `https://your-frontend.onrender.com` |

### Frontend service
| Field | Value |
|---|---|
| Build Command | `pip install -r requirements.txt` |
| Start Command | `streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0` |
| Env: `BACKEND_URL` | `https://your-backend.onrender.com` |

> Free tier services sleep after 15 min of inactivity — first request takes ~30s to wake up.

---

## 🛠️ Tech Stack

- **Backend:** FastAPI · Uvicorn · Pydantic v2 · joblib
- **Frontend:** Streamlit · Matplotlib · custom CSS eco theme
- **ML:** scikit-learn Pipeline + ColumnTransformer · XGBoost
- **Explainability:** SHAP (TreeExplainer / LinearExplainer)
- **Deployment:** Render
