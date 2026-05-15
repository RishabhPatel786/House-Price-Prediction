import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

# DIVERSE TRAINING DATA (Critical for Linear Regression accuracy)
# Features: [sqft, bhk, age, tier, dist_road, dist_metro, furnish, floor, type, road_width]
data = {
    'sqft': [600, 1000, 1200, 1800, 2500, 3500, 800, 1500, 3000, 5000, 700, 1200, 2500],
    'bhk':  [1,   2,    2,    3,    3,    4,    2,   3,    4,    5,    1,   2,    4],
    'age':  [20,  10,   5,    2,    1,    0,    15,  5,    2,    1,    12,  8,    3],
    'tier': [0,   0,    1,    1,    2,    2,    0,   1,    1,    2,    0,   1,    2], 
    'dist_road': [5.0, 1.0, 0.5, 0.2, 0.1, 0.0, 2.0, 0.5, 0.2, 0.05, 3.0, 0.8, 0.1],
    'dist_metro':[20,  10,  5,   2,   1,   0.5,  15,  8,   3,   0.2,  18,  7,   1],
    'furnish': [0,   1,    1,    2,    1,    2,    0,   1,    2,    2,    0,   1,    2],
    'floor': [1,   2,    5,    10,   15,   25,   1,   3,    12,   40,   1,   4,    18],
    'type':  [0,   0,    0,    1,    1,    2,    0,   0,    1,    2,    0,   0,    1],
    'road_width':[15,  25,   30,   40,   60,   100,  20,  35,   50,   150,  20,  30,   80],
    # PRICES (₹ Lakhs) - Manually balanced to ensure high variance
    'price': [8.5, 18.0, 45.0, 95.0, 210.0, 650.0, 14.0, 65.0, 145.0, 1500.0, 11.0, 52.0, 320.0]
}

df = pd.DataFrame(data)

# The Pipeline ensures inputs are scaled (normalized) before the Linear Math is applied
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LinearRegression())
])

model_pipeline.fit(df.drop('price', axis=1), df['price'])

# Save the updated brain
with open('model.pkl', 'wb') as f:
    pickle.dump(model_pipeline, f)

print("✅ BRAIN REBUILT: Model now understands price variance.")