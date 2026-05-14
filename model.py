import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

# Calibrated Dataset (2025-26 Indian Market)
data = {
    'sqft':      [600, 800, 1000, 1200, 1500, 1800, 2000, 2500, 3000, 3500, 4000],
    'beds':      [1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6],
    'baths':     [1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5],
    'tier':      [0, 0, 1, 1, 2, 2, 1, 2, 2, 2, 2],
    'prop_type': [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2],
    'furnish':   [0, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2],
    'price':     [8, 15, 45, 60, 110, 140, 95, 250, 380, 500, 650]
}

df = pd.DataFrame(data)
X = df.drop('price', axis=1)
y = df['price']

# Create and save Pipeline
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

model_pipeline.fit(X, y)
with open('model.pkl', 'wb') as f:
    pickle.dump(model_pipeline, f)

print("✅ Success: model.pkl created.")