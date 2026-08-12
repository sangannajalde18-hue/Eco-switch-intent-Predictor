"""
API request helpers for the Streamlit frontend.
"""

import os
from typing import Any, Optional

import requests
import streamlit as st

def _get_backend_url() -> str:
    """Resolve backend URL from secrets, env var, or default."""
    try:
        url = st.secrets.get("BACKEND_URL", None)
        if url:
            return url
    except Exception:
        pass
    return os.getenv("BACKEND_URL", "http://localhost:8000")

BACKEND_URL: str = _get_backend_url()


def get_health() -> Optional[dict]:
    """Ping the backend health endpoint. Returns None if unreachable."""
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return resp.json()
    except Exception:
        return None


def post_predict(payload: dict[str, Any]) -> dict[str, Any]:
    """
    POST the survey payload to /api/v1/predict.

    Returns a dict with keys:
      - "ok": bool
      - "data": dict (on success)
      - "error": str (on failure)
      - "status_code": int
    """
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/v1/predict",
            json=payload,
            timeout=30,
        )
        if resp.status_code == 200:
            return {"ok": True, "data": resp.json(), "status_code": 200}
        elif resp.status_code == 422:
            errors = resp.json().get("detail", [])
            msg = "; ".join(
                f"{' → '.join(str(loc) for loc in e.get('loc', []))}: {e.get('msg', '')}"
                for e in (errors if isinstance(errors, list) else [{"msg": str(errors)}])
            )
            return {"ok": False, "error": f"Validation error: {msg}", "status_code": 422}
        else:
            detail = resp.json().get("detail", "Unknown server error")
            return {"ok": False, "error": str(detail), "status_code": resp.status_code}
    except requests.exceptions.ConnectionError:
        return {
            "ok": False,
            "error": (
                f"Cannot reach the backend at **{BACKEND_URL}**. "
                "Make sure the FastAPI server is running."
            ),
            "status_code": 0,
        }
    except requests.exceptions.Timeout:
        return {
            "ok": False,
            "error": "The backend took too long to respond. Please try again.",
            "status_code": 0,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "status_code": 0}
