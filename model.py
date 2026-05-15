import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

# 2026 Professional Dataset - High-Density Samples
data = {
    'sqft':   [700, 1000, 1200, 1500, 2000, 2500, 3500, 700, 1200, 2000, 3000, 700, 1200, 2500, 4000],
    'beds':   [1,   2,    2,    3,    3,    4,    5,    1,   2,    3,    4,    1,   3,    4,    6],
    'baths':  [1,   1,    2,    2,    2,    3,    4,    1,   2,    2,    3,    1,   2,    3,    5],
    'tier':   [0,   0,    0,    0,    0,    0,    0,    1,   1,    1,    1,    2,   2,    2,    2],
    'type':   [0,   0,    0,    1,    1,    1,    2,    0,   1,    1,    2,    0,   1,    2,    2],
    'furnish':[0,   1,    1,    1,    1,    2,    2,    1,   1,    1,    2,    2,    2,    2,    2],
    # Prices (₹ Lakhs) - Forced clear separation for Linear Regression
    'price':  [15,  22,   30,   45,   65,   85,   140,  45,  78,   120,  210,  95,  180,  450,  1250]
}

df = pd.DataFrame(data)
X = df.drop('price', axis=1)
y = df['price']

# Professional Pipeline with Feature Scaling
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LinearRegression())
])

model_pipeline.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model_pipeline, f)

print("✅ Professional Brain Initialized.")