# Crop-Yield-Prediction-Using-Remote-Sensing-Weather-Data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Step 1: Load Dataset
# -----------------------------

data = pd.read_excel('crop_yield_data.xlsx')
print(data.head())

# -----------------------------
# Step 2: Train-Test Split
# -----------------------------
X = data.drop("Crop_Yield_t_ha", axis=1)
y = data["Crop_Yield_t_ha"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Step 3: Train ML Model
# -----------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Step 4: Predictions & Evaluation
# -----------------------------
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"RMSE: {rmse:.3f}")
print(f"R² Score: {r2:.3f}")

# -----------------------------
# Step 5: Visualization
# -----------------------------

# Actual vs Predicted
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred, color="green", alpha=0.7)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         "r--")
plt.xlabel("Actual Yield (t/ha)")
plt.ylabel("Predicted Yield (t/ha)")
plt.title("Actual vs Predicted Crop Yield")
plt.show()

# Feature importance
importances = model.feature_importances_
feat_importance = pd.Series(importances, index=X.columns)
feat_importance.sort_values().plot(kind='barh', figsize=(7,5), color="skyblue")
plt.title("Feature Importance in Crop Yield Prediction")
plt.show()
