# 🌾 Crop Yield Prediction Using Remote Sensing & Weather Data

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3d6b4a?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-e8b03a?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-c97a20?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-3d6b4a?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

**An end-to-end ML pipeline fusing satellite-derived vegetation indices with climate variables to forecast crop productivity — scalable to real-world Sentinel-2 & Landsat datasets.**

[🚀 Run Dashboard](https://crop-yield-prediction-using-remote-sensing-weather-data-jcs2k6.streamlit.app/) · [📊 View Results](#-model-results--visualizations) · [📐 Methodology](#-methodology) · [👤 Author](#-author)

</div>

---

## 📌 Project Overview

This project builds an integrated agricultural analytics pipeline that combines **vegetation health indices** (NDVI, EVI), **weather variables** (Rainfall, Temperature, Humidity), and **soil data** (Soil Moisture) to predict crop yield in tons per hectare (t/ha).

A **Random Forest Regressor** is used for its ability to capture non-linear environmental interactions and provide interpretable feature importance rankings. The framework is designed to scale seamlessly with real satellite datasets such as Sentinel-2 and Landsat.

### 🎯 Objectives

- Build an integrated agricultural dataset combining vegetation and climate indicators
- Develop a reproducible ML workflow for yield forecasting
- Evaluate model performance using robust regression metrics
- Interpret environmental drivers influencing yield
- Deliver a portfolio-ready environmental ML project

### 🌍 Applications

| Domain | Use Case |
|---|---|
| 🌱 Precision Agriculture | Targeted resource allocation based on predicted yield zones |
| 🌍 Climate-Smart Farming | Adaptive decisions under changing climate conditions |
| 📊 Food Security Planning | Data-driven national and regional food policy insights |
| 🛰 Remote Sensing Ops | Scalable to Sentinel-2, Landsat, and GIS-ready workflows |

---

## 📊 Key Results at a Glance

<div align="center">

| Metric | Value | Description |
|:---:|:---:|:---|
| 🎯 **R² Score** | **0.855** | 85.5% of yield variance explained |
| 📉 **RMSE** | **1.026 t/ha** | Average prediction error |
| 🗃 **Dataset** | **200 samples** | 6 environmental features |
| 🌾 **Mean Yield** | **6.67 t/ha** | Range: −2.3 to 12.2 t/ha |

</div>

---

## 🗂 Project Structure

```
Crop-Yield-Prediction-Using-Remote-Sensing-Weather-Data/
│
├── 📊 crop_yield_data.xlsx        # Synthetic dataset (200 samples, 6 features)
├── 🐍 crop_yield_prediction.py    # Core ML training & visualization script
├── 🖥  dashboard.py               # Streamlit interactive dashboard
├── 📋 requirements.txt            # Project dependencies
└── 📖 README.md                   # Documentation
```

---

## 📋 Dataset & Features

**Type:** Synthetic · **Samples:** 200 · **Purpose:** ML workflow demonstration

| Feature | Category | Description | Range | Importance |
|---|:---:|---|:---:|:---:|
| 🌡 **Temperature (°C)** | Weather | Surface air temperature | 15–40 °C | 🟩🟩🟩🟩🟩 53.6% |
| 🌧 **Rainfall (mm)** | Weather | Total precipitation | 100–1200 mm | 🟩🟩🟩 26.3% |
| 🛰 **NDVI** | Remote Sensing | Normalized Difference Vegetation Index | 0.20–0.89 | 🟨 8.8% |
| 💧 **Soil Moisture** | Soil | Root-zone water availability | 0.10–0.50 | 🟧 5.7% |
| 💦 **Humidity (%)** | Weather | Atmospheric moisture content | 30–90 % | ⬜ 3.0% |
| 🌿 **EVI** | Remote Sensing | Enhanced Vegetation Index | 0.10–0.79 | ⬜ 2.7% |
| **🌾 Crop Yield (t/ha)** | 🎯 **Target** | Crop productivity in tons per hectare | −2.3–12.2 | — |

---

## 📐 Methodology

```
┌─────────────────────────────────────────────────────────────┐
│                    ML PIPELINE WORKFLOW                     │
├──────────┬──────────┬──────────┬──────────────────────────┤
│  STEP 1  │  STEP 2  │  STEP 3  │         STEP 4           │
│    📥    │    🌲    │    📐    │           🔍             │
│  Data    │  Model   │  Evalu-  │      Interpretation      │
│  Prep    │ Training │  ation   │                          │
├──────────┼──────────┼──────────┼──────────────────────────┤
│ Load     │ Random   │ RMSE     │ Feature importance       │
│ Excel    │ Forest   │ & R²     │ ranking                  │
│ dataset  │ 100 trees│ scores   │                          │
│          │          │          │                          │
│ 80/20    │ Captures │ Scatter  │ Environmental yield      │
│ split    │ nonlinear│ plots    │ driver analysis          │
│          │ patterns │          │                          │
└──────────┴──────────┴──────────┴──────────────────────────┘
```

### 🌲 Why Random Forest?

- Handles **nonlinear** environmental relationships naturally
- **Robust** on tabular agricultural data
- Reduces overfitting through **ensemble learning**
- Provides built-in **feature importance** for interpretability

---

## 📊 Model Results & Visualizations

### 1️⃣ Feature Importance

```
Temperature (°C)  ████████████████████████████████████████  53.6%
Rainfall (mm)     ████████████████████                       26.3%
NDVI              ██████                                      8.8%
Soil Moisture     ████                                        5.7%
Humidity (%)      ██                                          3.0%
EVI               ██                                          2.7%
```

> **Key insight:** Temperature is the dominant predictor, followed by Rainfall — confirming that thermal and water stress are primary drivers of yield in this dataset.

---

### 2️⃣ Feature Correlation with Crop Yield

```
Rainfall (mm)     ████████████████  +0.416  (positive — more rain → higher yield)
NDVI              ██████████        +0.325  (positive — healthier vegetation → higher yield)
Soil Moisture     ████              +0.150  (positive — wetter soil → higher yield)
Humidity (%)      ▏                 +0.012  (negligible)
EVI               ████████         −0.122  (weakly negative)
Temperature (°C)  █████████████    −0.430  (negative — heat stress → lower yield)
```

---

### 3️⃣ Actual vs Predicted — Test Set

```
Yield   12 |        ×
(t/ha)  10 |  ·  ×      ×         ×    ×
         8 |    ·   ×       ×  ×     ×
         6 |  ×   ·    ·  ×       ·    ×
         4 |    ×    ·   ×    ·       ×
         2 |  ×    ·                ×
         0 | ×
        -2 |__________________________________
            0    5   10   15   20   25   30   40
                        Sample Index
                 · Actual   × Predicted
```

> Model tracks actual values closely across the full yield range. Larger deviations occur at yield extremes (very low or very high), a known limitation of ensemble methods on sparse edge cases.

---

### 4️⃣ Yield Distribution — All 200 Samples

```
Count
 31 |                          ████
 29 |                     ████ ████
 26 |                     ████ ████ ████
 22 |               ████  ████ ████ ████
 21 |          ████ ████  ████ ████ ████
 15 |          ████ ████  ████ ████ ████ ████
 14 |     ████ ████ ████  ████ ████ ████ ████
  9 | ████ ████ ████ ████ ████ ████ ████ ████ ████
  1 |─────────────────────────────────────────────────
     -2.3 -0.4  1.6  3.5  5.4  7.4  9.3 11.2
                      Yield (t/ha)
```

> Distribution is approximately normal, centred around 6–8 t/ha, with a slight left tail at negative yields (synthetic data artefact).

---

### 5️⃣ NDVI vs Crop Yield (Scatter)

```
Yield  12 |                    ·   ·
(t/ha) 10 |          ·   ·  ·   ·     · ·
        8 |    ·  ·     ·   ·  · ·  ·  ·  ·
        6 |  ·   ·  ·  ·  ·   ·    ·  ·  ·
        4 |  · ·   ·     ·  ·    ·  ·
        2 |    · ·   ·          ·
        0 |  ·
       -2 |__________________________________________
           0.20  0.30  0.40  0.50  0.60  0.70  0.80  0.90
                               NDVI
```

> A moderate positive trend (r = +0.325) — healthier vegetation (higher NDVI) generally correlates with higher yield, though Temperature and Rainfall remain stronger predictors.

---

## 💡 Key Insights

1. **Temperature is the primary driver** — accounting for 53.6% of feature importance; heat stress significantly suppresses yield
2. **Rainfall has strong secondary influence** — 26.3% importance and r = +0.416 with yield
3. **NDVI correlates positively** with yield (r = +0.325), confirming satellite vegetation monitoring is a valid yield proxy
4. **Ensemble learning outperforms** single-tree models by reducing variance across heterogeneous environmental data
5. **Integrating remote sensing with weather data** creates a more stable and interpretable prediction framework than either data source alone

---

## ▶️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Nelvinebi/Crop-Yield-Prediction-Using-Remote-Sensing-Weather-Data.git
cd Crop-Yield-Prediction-Using-Remote-Sensing-Weather-Data
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3a. Run the ML Script

```bash
python crop_yield_prediction.py
```

### 3b. Launch the Streamlit Dashboard

```bash
streamlit run dashboard.py
```

> ⚠️ Ensure `crop_yield_data.xlsx` is in the same directory as the script.

---

## 🛠 Tools & Technologies

| Tool | Version | Purpose |
|---|:---:|---|
| ![Python](https://img.shields.io/badge/-Python-3d6b4a?logo=python&logoColor=white) | 3.8+ | Core language |
| ![scikit-learn](https://img.shields.io/badge/-scikit--learn-e8b03a?logo=scikit-learn&logoColor=white) | 1.4+ | ML modelling & evaluation |
| ![Pandas](https://img.shields.io/badge/-Pandas-173350?logo=pandas&logoColor=white) | 2.0+ | Data manipulation |
| ![NumPy](https://img.shields.io/badge/-NumPy-3d6b4a?logo=numpy&logoColor=white) | 1.26+ | Numerical computing |
| ![Matplotlib](https://img.shields.io/badge/-Matplotlib-c97a20?logoColor=white) | 3.8+ | Static visualization |
| ![Plotly](https://img.shields.io/badge/-Plotly-255080?logo=plotly&logoColor=white) | 5.20+ | Interactive charts |
| ![Streamlit](https://img.shields.io/badge/-Streamlit-c97a20?logo=streamlit&logoColor=white) | 1.32+ | Web dashboard |
| ![OpenPyXL](https://img.shields.io/badge/-OpenPyXL-6fa87e?logoColor=white) | 3.1+ | Excel file I/O |

---

## ⚠️ Limitations

- Dataset is **synthetic** — not derived from real satellite imagery
- No **temporal sequence modelling** (crops are inherently seasonal)
- No **spatial GIS integration** or coordinate data
- Minimal **hyperparameter tuning** performed
- Single crop type — no multi-crop generalisation

---

## 🔮 Future Improvements

- [ ] Integrate real **Sentinel-2 / Landsat** satellite datasets
- [ ] Add **time-series modelling** with LSTM or Temporal CNN
- [ ] Perform **hyperparameter optimisation** via GridSearchCV / Optuna
- [ ] Deploy interactive **Streamlit / Flask web dashboard**
- [ ] GIS-based **spatial yield mapping** with GeoPandas & Folium
- [ ] **Multi-crop** dataset expansion and generalisation

---

## 👤 Author

<div align="center">

**AGBOZU EBINGIYE NELVIN**

*Environmental Data Scientist · GIS · Remote Sensing · Machine Learning*

End-to-end environmental intelligence solutions for flood risk, water quality, vegetation monitoring, and climate-resilient planning.

[![GitHub](https://img.shields.io/badge/GitHub-Nelvinebi-3d6b4a?style=for-the-badge&logo=github)](https://github.com/Nelvinebi)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-agbozu--ebi-255080?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/agbozu-ebi/)

</div>

---

<div align="center">

🌾 *"Bridging satellite intelligence and machine learning for a food-secure future."*

**MIT License** · © 2025 Agbozu Ebingiye Nelvin

</div>
