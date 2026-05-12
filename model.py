import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

# Calibrated 2026 Dataset
data = {
    'sqft':      [600, 1000, 1500, 2500, 800, 1200, 2000, 3000, 800, 1500, 4000, 1100],
    'beds':      [1, 2, 3, 4, 2, 3, 3, 4, 2, 3, 5, 2],
    'baths':     [1, 2, 2, 3, 1, 2, 3, 4, 2, 2, 5, 2],
    'tier':      [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 1], # 0:Rural, 1:Tier2, 2:Metro
    'prop_type': [0, 0, 1, 1, 0, 1, 1, 2, 0, 1, 2, 0],
    'furnish':   [0, 1, 1, 2, 1, 1, 2, 2, 1, 2, 2, 1],
    'price':     [7, 18, 38, 68, 32, 68, 115, 230, 88, 175, 680, 59] 
}

df = pd.DataFrame(data)

# Building a Pipeline (Standard practice for Linear Regression)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

pipeline.fit(df.drop('price', axis=1), df['price'])

with open('model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("✅ model.pkl created with Scaled Linear Regression.")