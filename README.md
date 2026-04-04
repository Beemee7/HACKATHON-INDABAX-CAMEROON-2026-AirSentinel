# HACKATHON-INDABAX-CAMEROON-2026-AirSentinel
Prédiction qualité de l'air à partir de données météorologiques dans le cadre du hackathon IndabaX Cameroon 2026

Pour ouvrir le notebook de la modélisation, cliquer ici
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Beemee7/HACKATHON-INDABAX-CAMEROON-2026-AirSentinel/blob/main/notebooks/IndabaX_Modelisation_github.ipynb)

# 🌍 AirCam AI — Prédiction de la Qualité de l'Air au Cameroun

> **IndabaX Cameroun 2026** · Thème : *L'IA au service de la résilience climatique et sanitaire au Cameroun*

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-brightgreen)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-brightgreen)](https://lightgbm.readthedocs.io)
[![Dash](https://img.shields.io/badge/Dashboard-Plotly%20Dash-purple)](https://dash.plotly.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📌 Présentation

**AirCam AI** est un système de prédiction de la concentration en PM2.5 (particules fines) pour **40 villes camerounaises** couvrant les 10 régions administratives du pays. Le projet combine des données météorologiques quotidiennes, de l'ingénierie de features avancée et des modèles de machine learning pour produire des prévisions à **7 jours**, sans nécessiter de capteurs terrain.

Le Cameroun ne dispose pas d'un réseau dense de stations de mesure de la qualité de l'air. Ce projet propose une approche basée sur des **données satellitaires (CAMS / Copernicus)** et des **prévisions météo open-source (Open-Meteo)** pour combler ce vide et informer les populations et décideurs sur les risques de pollution.

---

## 📊 Données

| Source | Description | Période |
|--------|-------------|---------|
| [Open-Meteo](https://open-meteo.com) / ERA5 | Variables météo journalières (température, vent, pluie, rayonnement…) | 2020–2025 |
| [Google Earth Engine](https://earthengine.google.com) — CAMS EAC4 | PM2.5 journalier par ville (réanalyse atmosphérique Copernicus) | 2020–2025 |
| [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api) | PM2.5 temps réel + prévisions (production) | Aujourd'hui → J+7 |

- **40 villes** · 4 par région · 10 régions administratives
- **~87 000 observations** journalières
- **Période** : 1er janvier 2020 → décembre 2025

---

## 🏗️ Architecture du Pipeline

```
Données météo (Open-Meteo / ERA5)
        +
Données PM2.5 (CAMS via GEE)
        │
        ▼
┌─────────────────────────────┐
│  Ingénierie des Features    │
│  • Encodage cyclique        │
│    (mois, jour, vent)       │
│  • Variables dérivées       │
│    (amplitude thermique,    │
│     fraction solaire…)      │
│  • Saison sèche / pluies    │
│    par région               │
│  • Lag PM2.5 (J-1)         │
│  • Target encoding (ville)  │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Sélection Lasso (TimeSeriesCV) │
│  31 → 25 features retenues  │
└────────────┬────────────────┘
             │
      ┌──────┴──────┐
      ▼             ▼
  Ridge CV      XGBoost / LightGBM / RF
  (modèle       (comparaison)
   retenu)
      │
      ▼
┌─────────────────────────────┐
│  Prédiction Itérative J+7  │
│  + Bias Calibration        │
└─────────────────────────────┘
             │
             ▼
     Dashboard Dash (Plotly)
     Chatbot (Ollama local)
```

---

## 🧪 Résultats

### Partie A — Modèles météo purs (sans persistance temporelle)

| Modèle | R² log | R² réel | RMSE réel |
|--------|--------|---------|-----------|
| Ridge | 0.587 | 0.467 | 13.4 µg/m³ |
| XGBoost | 0.577 | 0.435 | 13.7 µg/m³ |
| LightGBM | 0.574 | 0.444 | — |
| Random Forest | 0.566 | 0.429 | 13.8 µg/m³ |

> Les variables météo seules expliquent moins de 50% de la variance. La météo module la pollution sans en être la source directe.

### Partie B — Avec persistance temporelle (lag PM2.5 J-1)

| Modèle | R² log | R² réel | RMSE réel | MAE réel |
|--------|--------|---------|-----------|----------|
| **Ridge ✓** | **0.866** | **0.789** | **8.4 µg/m³** | **4.8 µg/m³** |
| Random Forest | 0.865 | 0.788 | 8.4 µg/m³ | 4.8 µg/m³ |
| XGBoost | 0.864 | 0.783 | 8.5 µg/m³ | 4.8 µg/m³ |
| LightGBM | 0.864 | ~0.784 | — | — |

> L'autocorrélation moyenne des PM2.5 est de **0.90** — la pollution de la veille domine le signal. Ridge atteint des performances quasi-identiques aux modèles complexes, avec une interprétabilité bien supérieure.

### Partie C — Prédiction itérative (simulation production réelle)

| Configuration | R² log | RMSE log |
|---------------|--------|----------|
| Itératif brut | 0.606 | 0.393 |
| **+ Bias calibration** | **0.627** | **0.382** |

> En production, le modèle ne connaît pas les vraies valeurs futures de PM2.5. Chaque prédiction est réinjectée comme lag pour le jour suivant. La **bias calibration** corrige la surestimation systématique (+0.017 log) et améliore le R² de +2 points.

---

## 🔍 Facteurs clés identifiés

D'après les coefficients Ridge standardisés :

| Facteur | Effet | Coefficient |
|---------|-------|-------------|
| PM2.5 veille (lag1) | ⬆️ Aggravant dominant | +0.587 |
| Température ressentie moyenne | ⬆️ Aggravant | +0.169 |
| Localisation (ville) | ⬆️ Aggravant (Nord) | +0.158 |
| Température 2m moyenne | ⬇️ Protecteur | -0.116 |
| Saison sèche | ⬆️ Aggravant | +0.038 |
| Précipitations | ⬇️ Protecteur (lessivage) | négatif |

**Message clé** : *la pollution de la veille domine, la saison sèche aggrave, la pluie améliore la qualité de l'air.*

---

## 🚀 Fonctionnalités du Dashboard

Le dashboard Plotly Dash (multi-pages) inclut :

- 🗺️ **Carte interactive** — niveaux PM2.5 en temps réel par ville
- 📈 **Prévisions J+7** — courbe de prédiction par ville avec catégorie AQI
- 🔔 **Système d'alertes** — seuils OMS 2021 avec code couleur
- 🤖 **Chatbot intégré** — assistant local via Ollama (open-source, sans coût API)
- 📊 **Analyse historique** — tendances 2020–2025 par région

---

## 📂 Structure du projet

```
HACKATHON-INDABAX-CAMEROON-2026-AirSentinel/
│
├── data/
│   ├── Dataset_complet_Meteo.csv       # Météo journalière 40 villes
│   └── donnes_qualite_air_journalier/  # PM2.5 CAMS (Google Earth Engine)
│
├── notebooks/
│   └── IndabaX_Modelisation_github.ipynb  # Pipeline ML

├── notebooks/
│   └── IndabaX_Modelisation.ipynb      # Pipeline ML complet
│
├── models/
│   ├── ridge_model.joblib              # Modèle Ridge entraîné
│   ├── target_encoder.joblib           # Encodeur ville
│   ├── scaler.joblib                   # StandardScaler
│   └── selected_features_lasso.json   # Features sélectionnées
│
├── dashboard/
│   ├── app.py                          # App Dash multi-pages
│   ├── pages/
│   │   ├── carte.py
│   │   ├── previsions.py
│   │   ├── historique.py
│   │   └── chatbot.py
│   └── assets/
│
├── scripts/
│   ├── download_pm25_openmeteo.py      # Téléchargement PM2.5 historique
│   └── openmeteo_forecast_j7.py       # Pipeline prédiction J+7
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
# Cloner le dépôt
git clone https://github.com/Beemee7/HACKATHON-INDABAX-CAMEROON-2026-AirSentinel.git
cd HACKATHON-INDABAX-CAMEROON-2026-AirSentinel

# Installer les dépendances
pip install -r requirements.txt

# Lancer le dashboard
python dashboard/app.py
```

**Dépendances principales :**
```
pandas, numpy, scikit-learn, xgboost, lightgbm
category_encoders, joblib
dash, plotly
requests, openmeteo-requests
ollama
```

---

## 🗓️ Données Open-Meteo en production

Le script `scripts/download_pm25_openmeteo.py` télécharge les PM2.5 historiques (2022 → aujourd'hui) pour les 40 villes via l'API gratuite CAMS d'Open-Meteo.

```python
# Téléchargement complet (~3 min pour 40 villes)
from scripts.download_pm25_openmeteo import download_all_cities
df = download_all_cities()
# → pm25_openmeteo_2020_today.csv (~87 000 lignes)
```

---

## 👥 Équipe

Projet réalisé dans le cadre du **Hackathon IndabaX Cameroun 2026** par le groupe **AirSentinel** coonstitué de:

| Nom | Responsabilité |
|------|---------------|
| Mlle BOUSA'A MERAWA| Modélisation, feature engineering, prédiction itérative(chef de groupe) |
| Mlle MATCHIM KOUOKAM Pierrette| Modélisation, feature engineering, prédiction itérative|
| M. SOKOUTCHOP SOKOUDJOU Divan Bryan| Frontend Dash, visualisations, chatbot |
| M. MBA-NZE Stéphane | Frontend Dash, visualisations, chatbot |

---


---

## 🙏 Remerciements

- [IndabaX Cameroun](https://indabaxcameroon.github.io) pour l'organisation du hackathon
- [Open-Meteo](https://open-meteo.com) pour les données météo et qualité de l'air open-source
- [Google Earth Engine](https://earthengine.google.com) pour l'accès aux données CAMS EAC4
