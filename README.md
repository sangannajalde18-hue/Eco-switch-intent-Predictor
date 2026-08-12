# 🌿 Eco-Switch Intent Predictor

Predicts whether a menstrual product user is likely to switch to an eco-friendly alternative,
based on a real-world survey of 333 respondents. Built with **FastAPI** + **Streamlit**.

---

## Project Structure

```
.
├── backend/
│   ├── __init__.py
│   ├── main.py        # FastAPI app (routes, CORS, lifespan)
│   ├── schemas.py     # Pydantic v2 request/response models
│   └── services.py    # Prediction logic + SHAP
├── frontend/
│   ├── __init__.py
│   ├── app.py         # Streamlit UI
│   ├── utils.py       # API call helpers
│   └── .streamlit/
│       └── secrets.toml
├── eco_switch_intent_pipeline.joblib  ← pre-trained model
├── requirements.txt
├── render.yaml
└── README.md
```

---

## Local Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the FastAPI backend
Open a terminal in the project root and run:
```bash
uvicorn backend.main:app --reload --port 8000
```
- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### 3. Run the Streamlit frontend
Open a **second terminal** in the project root and run:
```bash
streamlit run frontend/app.py
```
- App: http://localhost:8501

---

## Deployment on Render

1. Push this repo to GitHub (keep `eco_switch_intent_pipeline.joblib` in the repo).
2. Go to [render.com](https://render.com) → New → Blueprint → connect your repo.
3. Render will read `render.yaml` and create both services automatically.
4. After the backend deploys, copy its URL and update `BACKEND_URL` in the frontend service's environment variables on Render.

---

## Model Notes

- **Dataset**: 333 respondents, 24 survey features
- **Target**: Binary — intends to switch (1) vs. Maybe/No (0)
- **CV ROC-AUC**: 0.58–0.65 (expected for a small survey dataset)
- **Explainability**: SHAP values shown per prediction
