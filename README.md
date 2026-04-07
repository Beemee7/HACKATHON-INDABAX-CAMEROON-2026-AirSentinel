# HACKATHON-INDABAX-CAMEROON-2026-AirSentinel
Prédiction de la qualité de l'air à partir de données météorologiques dans le cadre du hackathon IndabaX Cameroon 2026

Pour ouvrir le notebook de la modélisation, cliquer ici :
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Beemee7/HACKATHON-INDABAX-CAMEROON-2026-AirSentinel/blob/main/notebooks/AirSentinel_notebook.ipynb)

# 🌬️ AirSentinel — Prédiction Multi-Cibles de la Qualité de l'Air au Cameroun

> **IndabaX Cameroun 2026** · Thème : *L'IA au service de la résilience climatique et sanitaire au Cameroun*

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-brightgreen)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-brightgreen)](https://lightgbm.readthedocs.io)
[![Dash](https://img.shields.io/badge/Dashboard-Plotly%20Dash-purple)](https://dash.plotly.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📌 Présentation

**AirCam** est un système de prédiction de la qualité de l'air pour **40 villes camerounaises** couvrant les 10 régions administratives du pays. Le projet combine des données météorologiques quotidiennes, de l'ingénierie de features avancée et des modèles de machine learning pour produire des prévisions **multi-cibles** sur 5 polluants atmosphériques simultanément : **PM2.5, NO₂, O₃, SO₂ et CO**.

Le Cameroun ne dispose pas d'un réseau dense de stations de mesure de la qualité de l'air. Ce projet propose une approche basée sur des **données satellitaires (CAMS / Copernicus)** et des **prévisions météo open-source (Open-Meteo)** pour combler ce vide et informer les populations et décideurs sur les risques de pollution.

> ⚠️ **PM2.5 reste le polluant le plus préoccupant** : 73,9 % des jours dépassent le seuil OMS (15 µg/m³), avec des pics atteignant 383 µg/m³ lors d'épisodes d'harmattan et de feux de brousse.

---

## 📊 Données

| Source | Description | Période |
|--------|-------------|---------|
| [Open-Meteo](https://open-meteo.com) / ERA5 | Variables météo journalières (température, vent, pluie, rayonnement…) | 2020–2025 |
| [Google Earth Engine](https://earthengine.google.com) — CAMS EAC4 | PM2.5, NO₂, O₃, SO₂, CO journaliers par ville (réanalyse Copernicus) | 2020–2025 |
| [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api) | Qualité de l'air temps réel + prévisions (production) | Aujourd'hui → J+7 |

- **40 villes** · 4 par région · 10 régions administratives
- **~87 200 observations** journalières
- **Période** : 1er janvier 2020 → décembre 2025

### Variables cibles

| Polluant | Unité | Conversion | Observations clés |
|----------|-------|------------|-------------------|
| **PM2.5** | µg/m³ | — | Moyenne 31 µg/m³ · max 383 µg/m³ · 73,9 % des jours > seuil OMS |
| **NO₂** | µg/m³ | kg/m³ × 10⁹ | Faible globalement · Yaoundé outlier (~1,46 µg/m³) |
| **O₃** | µg/m³ | kg/m³ × 10⁹ | Modéré · gradient inversé (Nord > Sud) · sous seuil OMS |
| **SO₂** | µg/m³ | kg/m³ × 10⁹ | Très faible · Limbe outlier (volcan + SONARA) |
| **CO** | µg/m³ | kg/m³ × 10⁹ | Dans les normes · combustion domestique diffuse |

---

## 🔍 Analyse Exploratoire — Points Clés

- **Gradient Nord–Sud pour PM2.5** : Kousseri (~44 µg/m³) vs Kribi (~21 µg/m³) — influence déterminante de l'harmattan
- **Saisonnalité marquée** : pic en saison sèche (feux + harmattan), creux en saison des pluies (lessivage atmosphérique)
- **Mémoire longue** : autocorrélation lag-1 de 0,73 à 0,93 selon le polluant — la pollution de la veille domine le signal
- **Distributions log-normales** : transformation logarithmique appliquée à tous les polluants avant modélisation
- **Aucune valeur manquante** dans le dataset fusionné — aucune imputation nécessaire
- **Outliers conservés** : ils correspondent à des épisodes de pollution réels (harmattan, éruptions, feux de brousse)

---

## 🏗️ Architecture du Pipeline

```
Données météo (Open-Meteo / ERA5)
        +
Données qualité de l'air (CAMS via GEE — 5 polluants)
        │
        ▼
┌──────────────────────────────────┐
│       Ingénierie des Features    │
│  • Transformations log par       │
│    polluant (log1p / log)        │
│  • Lags temporels J-1            │
│  • Encodage cyclique             │
│    (mois, jour, direction vent)  │
│  • Variables dérivées            │
│    (amplitude thermique,         │
│     fraction solaire…)           │
│  • Saison sèche / pluies         │
│    par région                    │
│  • Target encoding (ville)       │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  Sélection Lasso (TimeSeriesCV)  │
└──────────────┬───────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
   Ridge CV        XGBoost / LightGBM / RF
   (modèle         (comparaison)
    retenu)
       │
       ▼
┌──────────────────────────────────┐
│  Prédiction Itérative J+7       │
│  + Bias Calibration             │
└──────────────────────────────────┘
               │
               ▼
      Dashboard Plotly Dash
      Chatbot (Ollama local)
```

---

## 🧪 Résultats de Modélisation

> Résultats présentés pour **PM2.5** (polluant prioritaire). Les modèles multi-cibles suivent la même architecture.

### Partie A — Modèles météo purs (sans persistance temporelle)

| Modèle | R² log | R² réel | RMSE réel |
|--------|--------|---------|-----------|
| Ridge | 0.587 | 0.467 | 13.4 µg/m³ |
| XGBoost | 0.577 | 0.435 | 13.7 µg/m³ |
| LightGBM | 0.574 | 0.444 | — |
| Random Forest | 0.566 | 0.429 | 13.8 µg/m³ |

> Les variables météo seules expliquent moins de 50 % de la variance — la météo module la pollution sans en être la source directe.

### Partie B — Avec persistance temporelle (lag PM2.5 J-1)

| Modèle | R² log | R² réel | RMSE réel | MAE réel |
|--------|--------|---------|-----------|----------|
| **Ridge ✓** | **0.866** | **0.789** | **8.4 µg/m³** | **4.8 µg/m³** |
| Random Forest | 0.865 | 0.788 | 8.4 µg/m³ | 4.8 µg/m³ |
| XGBoost | 0.864 | 0.783 | 8.5 µg/m³ | 4.8 µg/m³ |
| LightGBM | 0.864 | ~0.784 | — | — |

> L'ajout du lag J-1 fait passer le R² de ~0.47 à **0.79** (+32 pts). Ridge atteint des performances quasi-identiques aux modèles complexes, avec une interprétabilité bien supérieure.

### Partie C — Prédiction itérative (simulation production réelle J+7)

| Configuration | R² log | RMSE log |
|---------------|--------|----------|
| Itératif brut | 0.606 | 0.393 |
| **+ Bias calibration** | **0.627** | **0.382** |

> En production, le modèle réinjecte ses propres prédictions comme lag pour les jours suivants. La **bias calibration** corrige la surestimation systématique (+0.017 log) et améliore le R² de +2 points.

---

## 🔑 Facteurs Clés Identifiés

D'après les coefficients Ridge standardisés :

| Facteur | Effet | Coefficient |
|---------|-------|-------------|
| PM2.5 veille (lag1) | ⬆️ Aggravant dominant | +0.587 |
| Température ressentie moyenne | ⬆️ Aggravant | +0.169 |
| Localisation (ville — Nord) | ⬆️ Aggravant | +0.158 |
| Température 2m moyenne | ⬇️ Protecteur | -0.116 |
| Saison sèche | ⬆️ Aggravant | +0.038 |
| Précipitations | ⬇️ Protecteur (lessivage) | négatif |

**Message clé** : *la pollution de la veille influence grandement celle d'aujourd'hui. La saison sèche aggrave la pollution, la pluie protège.*

---

## 🖥️ Dashboard Interactif

> 🚧 **En cours de développement** — La description complète sera ajoutée ici prochainement.

Le dashboard Plotly Dash (multi-pages) inclura :

- 🗺️ **Carte interactive** — niveaux des 5 polluants en temps réel par ville
- 📈 **Prévisions J+7** — courbe de prédiction par ville avec catégorie AQI
- 🔔 **Système d'alertes** — seuils OMS 2021 avec code couleur, messagerie automatique
- 🤖 **Chatbot intégré** — assistant local via Ollama (open-source, sans coût API)
- 📊 **Analyse historique** — tendances 2020–2025 par région

---

## 📂 Structure du Projet

```
HACKATHON-INDABAX-CAMEROON-2026-AirSentinel/
│
├── data/
│   ├── Dataset_complet_Meteo.csv            # Météo journalière 40 villes
│   └── donnes_qualite_air_journalier.csv    # 5 polluants CAMS (GEE)
│
├── notebooks/
│   └── AirSentinel_notebook.ipynb           # Pipeline complet (EDA + modélisation)
│
├── models/
│   ├── ridge_model.joblib                   # Modèle Ridge entraîné
│   ├── target_encoder.joblib                # Encodeur ville
│   ├── scaler.joblib                        # StandardScaler
│   └── selected_features_lasso.json        # Features sélectionnées
│
├── dashboard/                              
│   ├── app.py
│   ├── pages/
│   │   ├── carte.py
│   │   ├── previsions.py
│   │   ├── historique.py
│   │   └── chatbot.py
│   └── assets/
│
├── scripts/
│   ├── download_pm25_openmeteo.py           # Téléchargement historique
│   └── openmeteo_forecast_j7.py            # Pipeline prédiction J+7
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
statsmodels, pmdarima, category_encoders, joblib
dash, plotly, requests, openmeteo-requests, ollama
```

---

## 👥 Équipe

Projet réalisé dans le cadre du **Hackathon IndabaX Cameroun 2026** par le groupe **AirSentinel** :

| Nom | Responsabilité |
|-----|----------------|
| Mlle BOUSA'A MERAWA *(Chef de groupe)* | Modélisation, feature engineering, prédiction itérative |
| Mlle MATCHIM KOUOKAM Pierrette | Modélisation, feature engineering, prédiction itérative |
| M. SOKOUTCHOP SOKOUDJOU Divan Bryan | Frontend Dash, visualisations, chatbot |
| M. MBA-NZE Stéphane | Frontend Dash, visualisations, chatbot |

---

## 🙏 Remerciements

- [IndabaX Cameroun](https://indabaxcameroon.github.io) pour l'organisation du hackathon
- [Open-Meteo](https://open-meteo.com) pour les données météo et qualité de l'air open-source
- [Google Earth Engine](https://earthengine.google.com) pour l'accès aux données CAMS EAC4
