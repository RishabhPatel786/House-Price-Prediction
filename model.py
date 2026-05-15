import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

# Calibrated 2026 Dataset - High Variance for Location
# tier: 0 (Rural), 1 (City), 2 (Metro)
data = {
    'sqft':   [700, 1200, 2500, 700, 1200, 2500, 700, 1200, 2500, 1500, 4000],
    'beds':   [2,   3,    4,    2,   3,    4,    2,   3,    4,    3,    5],
    'baths':  [1,   2,    3,    2,   2,    3,    2,   3,    3,    3,    5],
    'tier':   [0,   0,    0,    1,   1,    1,    2,   2,    2,    1,    2],
    'type':   [0,   0,    1,    0,   0,    1,    0,   1,    2,    1,    2],
    'furnish':[0,   1,    1,    1,   1,    1,    1,   2,    2,    1,    2],
    # Aggressive pricing gaps (in Lakhs) to force the model to learn locality impact
    'price':  [12,  24,   45,   48,  82,   165,  115, 240,  650,  110,  1200]
}

df = pd.DataFrame(data)
X = df.drop('price', axis=1)
y = df['price']

# Pipeline ensures Tier and Sqft are analyzed on the same mathematical scale
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LinearRegression())
])

model_pipeline.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model_pipeline, f)

print("✅ BRAIN RECALIBRATED: Locality weight increased.")