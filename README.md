🌾 Crop Yield Prediction Using Remote Sensing & Weather Data


🚀 Project Overview

This project develops a machine learning pipeline for crop yield prediction by integrating synthetic remote sensing indices (NDVI, EVI, Soil Moisture) with weather variables (Rainfall, Temperature, Humidity).

The goal is to demonstrate how satellite-derived indicators and climate data can be combined to support:

🌍 Climate-smart agriculture

🌾 Precision farming

📊 Data-driven food security planning

This project reflects a scalable framework that can be adapted to real satellite datasets (e.g., Sentinel-2, Landsat) for operational agricultural analytics.

🎯 Objectives

Build an integrated agricultural dataset combining vegetation and climate indicators

Develop a reproducible ML workflow for yield forecasting

Evaluate model performance using robust regression metrics

Interpret environmental drivers influencing yield

Provide a portfolio-ready environmental ML project

🖼️ Project Outputs
📊 1. Actual vs Predicted Yield

Scatter plot validating model performance.

📈 2. Feature Importance Ranking

Bar chart showing dominant environmental predictors.

📉 3. Model Metrics

RMSE

R² Score

These outputs demonstrate interpretability and predictive capability.

🗂️ Project Structure
Crop-Yield-Prediction-Using-Remote-Sensing-Weather-Data/
│
├── crop_yield_data.xlsx        # Synthetic dataset (>200 samples)
├── crop_yield_prediction.py    # Model training & visualization script
├── requirements.txt            # Project dependencies
└── README.md                   # Documentation
📊 Dataset Description

Type: Synthetic
Samples: >200
Purpose: ML workflow demonstration

🔎 Features
Feature	Description
NDVI	Vegetation health index
EVI	Enhanced vegetation index
Rainfall (mm)	Precipitation level
Temperature (°C)	Surface temperature
Humidity (%)	Atmospheric moisture
Soil Moisture	Root-zone water availability
🎯 Target Variable

Crop Yield (tons/hectare)

The dataset simulates realistic environmental-yield relationships.

🔬 Methodology
1️⃣ Data Preparation

Load Excel dataset

Feature-target separation

Train-test split

2️⃣ Modeling

Train Random Forest Regressor

Capture nonlinear environmental interactions

3️⃣ Evaluation

Compute RMSE & R²

Generate validation plots

4️⃣ Interpretation

Extract feature importance

Analyze environmental yield drivers

🤖 Models & Techniques
🌲 Random Forest Regressor

Chosen because it:

Handles nonlinear relationships

Performs well on tabular environmental data

Reduces overfitting through ensemble learning

Provides built-in feature importance

📈 Model Evaluation & Results
Metric	Purpose
RMSE	Measures prediction error magnitude
R² Score	Measures explained variance

The model demonstrates strong predictive consistency across synthetic environmental conditions.

💡 Key Insights

Vegetation indices (NDVI, EVI) strongly influence yield

Soil moisture and rainfall significantly affect productivity

Ensemble learning improves agricultural forecasting reliability

Integrating remote sensing with weather data enhances prediction stability

🌍 Applications

Precision agriculture systems

Agricultural risk modeling

Climate adaptation planning

Crop monitoring dashboards

Food security analysis

🛠️ Tools & Technologies

Python 3.8+

NumPy

Pandas

Matplotlib

Scikit-learn

OpenPyXL

▶️ How to Run
1️⃣ Clone the Repository
git clone https://github.com/Nelvinebi/Crop-Yield-Prediction-Using-Remote-Sensing-Weather-Data.git
cd Crop-Yield-Prediction-Using-Remote-Sensing-Weather-Data
2️⃣ Install Dependencies
pip install -r requirements.txt

Or manually:

pip install numpy pandas matplotlib scikit-learn openpyxl
3️⃣ Run the Script
python crop_yield_prediction.py

Ensure crop_yield_data.xlsx is in the same directory.

⚠️ Limitations

Dataset is synthetic

No temporal sequence modeling

No spatial GIS integration

Minimal hyperparameter tuning

🔮 Future Improvements

Integrate real satellite datasets (Sentinel-2, Landsat)

Add time-series modeling (LSTM, Temporal CNN)

Perform hyperparameter optimization

Deploy as a web dashboard (Streamlit / Flask)

Integrate GIS-based spatial yield mapping

👤 Author
AGBOZU EBINGIYE NELVIN

Environmental Data Scientist | GIS | Remote Sensing | Machine Learning

I develop end-to-end, GIS-ready environmental intelligence solutions for flood risk, water quality, vegetation monitoring, and climate-resilient planning.

🔗 GitHub: https://github.com/Nelvinebi

🔗 LinkedIn: https://www.linkedin.com/in/agbozu-ebi/

