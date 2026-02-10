Crop-Yield-Prediction-Using-Remote-Sensing-Weather-Data

This project demonstrates a machine learning approach to predict crop yield using synthetic remote sensing data (NDVI, EVI, Soil Moisture) and weather data (Rainfall, Temperature, Humidity). It helps illustrate how agricultural datasets can be combined with machine learning for yield forecasting.

📂 Project Structure

crop_yield_data.xlsx → Synthetic dataset (>200 samples).

crop_yield_prediction.py → Python script for model training, evaluation, and visualization.

README.md → Project documentation.

⚙️ Features

Generates and uses synthetic crop yield dataset.

Trains a Random Forest Regressor for yield prediction.

Evaluates performance with RMSE and R² Score.

Visualizes:

Actual vs Predicted yield.

Feature importance ranking.

🚀 How to Run

Clone or download this repository.

Ensure you have Python 3.8+ installed.

Install dependencies:

pip install -r requirements.txt


(or manually: numpy pandas matplotlib scikit-learn openpyxl)

Place crop_yield_data.xlsx in the same folder as crop_yield_prediction.py.

Run the script:

python crop_yield_prediction.py

📊 Dataset

The dataset is synthetic and includes:

NDVI (Normalized Difference Vegetation Index)

EVI (Enhanced Vegetation Index)

Rainfall (mm)

Temperature (°C)

Humidity (%)

Soil Moisture

Crop Yield (target variable, tons/ha)

📈 Example Output

Model Performance Metrics (RMSE, R² Score).

Scatter plot of Actual vs Predicted yield.

Bar chart of feature importance.

🔮 Future Improvements

Replace synthetic data with real satellite & weather datasets.

Experiment with deep learning models (e.g., LSTM, CNN) for time-series yield prediction.

Integrate with GIS platforms for spatial yield mapping.

Author Name: AGBOZU EBINGIYE NELVIN
Github: https://github.com/Nelvinebi
LinkedIn: https://www.linkedin.com/in/agbozu-ebi/
