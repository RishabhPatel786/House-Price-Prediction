import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
import pickle
import json

# ============================================================
# Advanced Indian Real Estate Dataset
# ============================================================

data = {
    'sqft': [
        600, 800, 1000, 1200, 1500, 1800, 2000,
        2500, 3000, 3500, 4000, 4500, 5000
    ],

    'beds': [
        1,2,2,3,3,3,4,4,5,5,6,6,7
    ],

    'baths': [
        1,1,2,2,2,3,3,4,4,5,5,6,6
    ],

    'tier': [
        0,0,1,1,2,2,1,2,2,2,2,2,2
    ],

    'prop_type': [
        0,0,0,1,1,1,1,2,2,2,2,2,2
    ],

    'furnish': [
        0,1,1,1,2,2,1,2,2,2,2,2,2
    ],

    'price': [
        8,15,45,60,110,140,95,250,380,500,650,800,950
    ]
}

df = pd.DataFrame(data)

# ============================================================
# Feature Engineering
# ============================================================

df['luxury_score'] = (
    (df['tier'] * 2) +
    (df['prop_type'] * 2) +
    df['furnish']
)

df['room_density'] = (
    df['sqft'] /
    (df['beds'] + df['baths'])
)

# ============================================================
# Features & Target
# ============================================================

X = df.drop('price', axis=1)
y = df['price']

# ============================================================
# Advanced Polynomial Linear Regression
# ============================================================

model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(
        degree=2,
        include_bias=False
    )),
    ('regressor', LinearRegression())
])

# Train
model_pipeline.fit(X, y)

# Predictions
y_pred = model_pipeline.predict(X)

# Metrics
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)

# ============================================================
# Save Model
# ============================================================

with open('model.pkl', 'wb') as f:
    pickle.dump(model_pipeline, f)

# Save Metrics
metrics = {
    "r2_score": round(r2, 4),
    "mae": round(mae, 2),
    "samples": len(df)
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f)

print("\n✅ Advanced Linear Regression Model Trained")
print(f"📊 R² Score      : {r2:.4f}")
print(f"📉 MAE Error     : {mae:.2f} Lakhs")
print(f"📦 Dataset Size  : {len(df)}")
print("🚀 model.pkl saved successfully")