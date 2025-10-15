# Imports des bibliothèques nécessaires
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score, classification_report, confusion_matrix, roc_curve)
import re
import string
import warnings
from typing import List, Dict
import time
import joblib
import os
from pydantic import BaseModel, Field
from datetime import datetime
import nltk
nltk.download('wordnet')    # Pour la lemmatisation (forme de base)



# test_api.py
import time
import math
import pytest
from fastapi.testclient import TestClient

# On importe l'app depuis main.py
import main

client = TestClient(main.app)

# --------- Helpers ---------
def _health():
    r = client.get("/health")
    assert r.status_code == 200
    return r.json()

def _approx_equal(a: float, b: float, tol: float = 1e-2) -> bool:
    return abs(a - b) <= tol

# --------- Tests ---------
def test_health_ok():
    """Vérifie que l’API répond et expose les infos de base."""
    h = _health()
    assert h["status"] == "ok"
    assert "model_loaded" in h and "vectorizer_loaded" in h
    assert "server_time" in h

def test_predict_endpoint_valid():
    """Test /predict avec un texte valide."""
    h = _health()
    if not h.get("model_loaded", False):
        pytest.skip("Modèle non chargé — on skip le test /predict valid.")

    payload = {"text": "J'adore ce produit, il est fantastique !"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200

    data = r.json()
    # Champs attendus
    for field in ["sentiment", "confidence", "probability_positive", "probability_negative"]:
        assert field in data

    # Types & bornes
    assert data["sentiment"] in ["Positif", "Négatif"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert 0.0 <= data["probability_positive"] <= 1.0
    assert 0.0 <= data["probability_negative"] <= 1.0

    # Somme des probabilités ≈ 1 (tolérance 1e-2)
    total = data["probability_positive"] + data["probability_negative"]
    assert _approx_equal(total, 1.0, 1e-2)

def test_predict_endpoint_invalid():
    """Test /predict avec des données invalides."""
    # Cas 1 : JSON vide -> 422
    r = client.post("/predict", json={})
    assert r.status_code == 422

    # Cas 2 : texte trop long -> 422 (max_length=280)
    long_text = "a" * 300
    r = client.post("/predict", json={"text": long_text})
    assert r.status_code == 422

    # Cas 3 : texte vide
    # Si TweetRequest a min_length=1 → 422 ; sinon l’API peut accepter et renvoyer 200.
    r = client.post("/predict", json={"text": ""})
    if r.status_code == 200:
        # Mode tolérant : on vérifie la structure de réponse
        data = r.json()
        for field in ["sentiment", "confidence", "probability_positive", "probability_negative"]:
            assert field in data
    else:
        assert r.status_code == 422

def test_explain_endpoint():
    """Test /explain avec LIME (skip si indisponible)."""
    h = _health()
    if not h.get("lime_available", False):
        pytest.skip("LIME indisponible sur ce serveur — skip /explain.")

    if not h.get("model_loaded", False):
        pytest.skip("Modèle non chargé — skip /explain.")

    payload = {"text": "Ce film est absolument terrible, je le déteste !"}
    t0 = time.time()
    r = client.post("/explain", json=payload)
    dt = time.time() - t0

    assert r.status_code == 200
    data = r.json()

    # Champs
    for field in ["sentiment", "explanation", "html_explanation"]:
        assert field in data

    # Explications non vides
    assert isinstance(data["explanation"], list)
    assert len(data["explanation"]) > 0

    # HTML substantiel
    html_content = data["html_explanation"]
    assert isinstance(html_content, str)
    assert len(html_content) > 100
    assert "<div" in html_content

    # Temps raisonnable (< 120s)
    assert dt < 120.0

@pytest.mark.parametrize("text", [
    "Super !",               # très court
    "😊" * 10,               # emojis uniquement
    "http://example.com ok"  # avec URL
])
def test_explain_robustness(text):
    """Robustesse /explain sur des textes pathologiques (skip si LIME indisponible)."""
    h = _health()
    if not h.get("lime_available", False):
        pytest.skip("LIME indisponible — skip robustesse.")
    if not h.get("model_loaded", False):
        pytest.skip("Modèle non chargé — skip robustesse.")

    r = client.post("/explain", json={"text": text})
    # Selon le prétraitement interne, certains cas extrêmes peuvent être rejetés par Pydantic (422).
    assert r.status_code in (200, 422)

    if r.status_code == 200:
        data = r.json()
        assert "html_explanation" in data