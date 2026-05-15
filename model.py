import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
import pickle

# 2026 Realistic Indian Market Data (Extreme Gaps)
# tier: 0 (Rural), 1 (Tier-2), 2 (Metro)
data = {
    'sqft':   [800, 1500, 3000, 800, 1500, 3000, 800, 1500, 3000, 5000],
    'beds':   [2,   3,    4,    2,   3,    4,    2,   3,    4,    5],
    'baths':  [1,   2,    3,    1,   2,    3,    2,   2,    3,    5],
    'tier':   [0,   0,    0,    1,   1,    1,    2,   2,    2,    2],
    'type':   [0,   0,    1,    0,   1,    1,    0,   1,    2,    2],
    'furnish':[0,   1,    1,    1,   1,    1,    1,   2,    2,    2],
    # PRICES (₹ Lakhs) - Forced clear mathematical separation
    'price':  [12,  22,   45,   45,  85,   170,  95,  210,  650,  1800] 
}

df = pd.DataFrame(data)

# Advanced Pipeline: Scaling + Polynomial Features + Regression
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)), # Allows non-linear price jumps
    ('lr', LinearRegression())
])

model_pipeline.fit(df.drop('price', axis=1), df['price'])

with open('model.pkl', 'wb') as f:
    pickle.dump(model_pipeline, f)

print("✅ BRAIN UPDATED: Non-linear price scaling active.")