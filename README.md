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

**AirSentinel** est un système de prédiction de la qualité de l'air pour **40 villes camerounaises** couvrant les 10 régions administratives du pays. Le projet combine des données météorologiques quotidiennes (Open‑Meteo / ERA5), de l’ingénierie de features avancée et des modèles de machine learning pour produire des prévisions **multi‑cibles** simultanées sur 5 polluants atmosphériques : **PM2.5, NO₂, O₃, SO₂ et CO**.

Le Cameroun ne dispose pas d’un réseau dense de stations de mesure. Ce projet comble ce vide en utilisant des **données satellitaires (CAMS reanalysis via Google Earth Engine)** et des **prévisions météo open‑source** pour informer les populations et les décideurs sur les risques de pollution.

> ⚠️ **PM2.5 reste le polluant le plus préoccupant** : moyenne nationale de 31 µg/m³, 73,9 % des jours dépassent le seuil OMS (15 µg/m³), avec des pics atteignant 383 µg/m³ lors d’épisodes d’harmattan et de feux de brousse.

---

## 📊 Données

| Source                          | Description                                      | Période       |
|---------------------------------|--------------------------------------------------|---------------|
| Open‑Meteo / ERA5               | Variables météo journalières (température, vent, pluie, rayonnement…) | 2020–2025 |
| Google Earth Engine — CAMS EAC4 | PM2.5, NO₂, O₃, SO₂, CO journaliers par ville   | 2020–2025 |
| Open‑Meteo Air Quality API      | Qualité de l’air temps réel + prévisions         | Aujourd’hui → J+7 |

- **40 villes** · 4 par région · 10 régions administratives  
- **87 200 observations** journalières  
- **Période** : 1er janvier 2020 → 19 décembre 2025

### Variables cibles
| Polluant | Unité cible | Conversion appliquée          | Observations clés |
|----------|-------------|-------------------------------|-------------------|
| **PM2.5** | µg/m³      | Aucune                        | Moyenne 31 µg/m³ · max 383 µg/m³ · 73,9 % > OMS |
| **NO₂**   | µg/m³      | kg/m³ → µg/m³ (×10⁹)          | Faible · Yaoundé outlier |
| **O₃**    | µg/m³      | kg/m³ → µg/m³ (×10⁹)          | Gradient Nord > Sud |
| **SO₂**   | µg/m³      | kg/m³ → µg/m³ (×10⁹)          | Limbe outlier (volcan + SONARA) |
| **CO**    | µg/m³      | kg/m³ → µg/m³ (×10⁹)          | Dans les normes · combustion domestique |

**Transformations logarithmiques** appliquées avant modélisation :
- PM2.5 → `log(1 + valeur)`
- NO₂, O₃, SO₂, CO → `log(valeur)`

---

## 🔍 Analyse Exploratoire — Points Clés

- **Aucune valeur manquante** dans le dataset fusionné.
- **Gradient Nord–Sud très marqué** pour PM2.5 (Kousseri ~44 µg/m³ vs Kribi ~21 µg/m³).
- **Saisonnalité forte** : pic en saison sèche (harmattan + feux), creux en saison des pluies.
- **Distributions log‑normales** → transformation log justifiée.
- **Outliers conservés** .
- **Stationnarité** : 4 polluants sur 5 stationnaires selon ADF + KPSS ; PM2.5 trend‑stationary.
- **Corrélation** : lag‑1 très élevé (0,73–0,93) → persistance temporelle dominante.

---

## 🏗️ Architecture du Pipeline

```text
Données météo + Qualité de l’air (CAMS)
│
▼
Ingénierie des features
├─ Lags temporels J-1 (chaque polluant)
├─ Variables cycliques (mois, jour, vent)
├─ Indicateurs dérivés (amplitude thermique, fraction solaire…)
├─ Saison sèche / pluies par région
└─ Target encoding de la ville
│
▼
Sélection de features (Lasso + TimeSeriesCV)
│
▼
Modèles ML multi‑cibles
• Ridge • Random Forest • XGBoost • LightGBM
│
▼
Prédiction itérative (horizon J+7) + calibration de biais
│
▼
Dashboard interactif (Plotly Dash + chatbot)
```

---

## 🧪 Résultats de Modélisation (Multi-Cibles)

Les 5 polluants sont prédits **simultanément** avec `MultiOutputRegressor`.  
Tous les modèles ont été entraînés **avec et sans lags**, en validation temporelle (`TimeSeriesSplit`).

### Partie A — Modèles météo purs (sans persistance temporelle)

| Modèle        | R² log (moyenne) | R² réel (moyenne) | RMSE réel moyen (µg/m³) |
|---------------|------------------|-------------------|--------------------------|
| Ridge         | 0.60             | 0.47              | 9.58                     |
| XGBoost       | 0.73             | 0.67              | 9.41                     |
| LightGBM      | 0.73             | 0.67              | 9.38                     |
| Random Forest | 0.72             | 0.67              | 9.40                     |

> **Observation** : sans le lag, la météo seule explique ~45 % de la variance → insuffisant.

### Partie B — Avec persistance temporelle (lag J-1 de chaque polluant)

| Modèle            | R² log moyen | R² réel moyen | RMSE réel moyen (µg/m³) | MAE réel moyen (µg/m³) |
|-------------------|--------------|---------------|-------------------------|------------------------|
| **LightGBM** ✓    | 0.86         | **0.8535**    | 5.54                    | 3.88                   |
| XGBoost           | 0.86         | 0.8536        | 5.54                    | 3.88                   |
| Random Forest     | 0.86         | 0.8522        | 5.54                    | 3.88                   |
| Ridge             | 0.86         | 0.8384        | 5.48                    | 3.83                   |

> **Gain énorme** : +32 à +34 points de R² grâce au lag J-1.  
> **LightGBM** est retenu comme modèle de production (rapide, mémoire réduite, stable en prédiction itérative).

#### Performances détaillées par polluant (LightGBM + lag J-1)

| Polluant | R² log | R² réel | RMSE réel (µg/m³) | MAE réel (µg/m³) | MAPE (%) |
|----------|--------|---------|---------------------|-------------------|----------|
| PM2.5    | 0.8634 | 0.7842  | 8.50                | 4.86              | 19.2     |
| NO₂      | 0.9140 | 0.8593  | 0.18                | 0.10              | 21.0     |
| O₃       | 0.9050 | 0.9286  | 3.34                | 2.57              | 8.5      |
| SO₂      | 0.9090 | 0.8499  | 0.07                | 0.04              | 25.9     |
| CO       | 0.8427 | 0.8455  | 15.59               | 11.80             | 7.4      |

> **Conclusion** : LightGBM domine sur les 5 cibles. O₃ est le plus facile à prédire (R² = 0.93). PM2.5 reste le plus difficile à cause de ses pics extrêmes.

### Partie C — Prédiction itérative (horizon J+7)

- Chaîne de prédictions récursive : chaque jour prédit devient l’input du jour suivant.
- Calibration de biais appliquée sur les 7 jours.
- Le modèle reste stable jusqu’à J+4 / J+5, puis dégradation progressive (classique pour les modèles avec lag).
---

## 🔑 Facteurs Clés Identifiés (importance LightGBM)

| Facteur                        | Effet          | Importance (gain) |
|--------------------------------|----------------|-------------------|
| PM2.5 veille (lag-1)           | ⬆️ Dominant    | très élevée       |
| Saison sèche                   | ⬆️ Aggravant   | élevée            |
| Température moyenne            | ⬆️ Aggravant   | modérée           |
| Précipitations                 | ⬇️ Protecteur  | modérée           |
| Rayonnement solaire            | ⬆️ (pour O₃)   | élevée            |
| Direction du vent (harmattan)  | ⬆️ Aggravant   | modérée           |

**Message clé** : la pollution de la veille est le facteur le plus déterminant. La saison sèche aggrave fortement la pollution ; la pluie la réduit (lessivage).

---

## 🖥️ Dashboard Interactif

Le dashboard Plotly Dash (multi-pages) inclura :
- 🗺️ **Carte interactive** — niveaux des 5 polluants en temps réel par ville
- 📈 **Prévisions J+7** — courbes de prédiction par ville avec catégorie AQI et seuils OMS
- 🔔 **Système d’alertes** — notifications selon les seuils OMS 2021
- 🤖 **Chatbot intégré** — assistant conversationnel via Ollama (open-source, sans coût API)
- 📊 **Analyse historique** — tendances 2020–2025 par région et par polluant

---




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

