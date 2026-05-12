import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

# Calibrated Data: High variance between Tiers to help the Linear Model "see" the difference
data = {
    'sqft':      [600, 1000, 1500, 2500, 800, 1200, 2000, 3000, 800, 1500, 4000],
    'beds':      [1, 2, 3, 4, 2, 3, 3, 4, 2, 3, 5],
    'baths':     [1, 2, 2, 3, 1, 2, 3, 4, 2, 2, 5],
    'tier':      [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2], # 0:Rural, 1:Tier2, 2:Metro
    'prop_type': [0, 0, 1, 1, 0, 1, 1, 2, 0, 1, 2],
    'furnish':   [0, 1, 1, 2, 1, 1, 2, 2, 1, 2, 2],
    'price':     [6, 18, 35, 65, 30, 65, 110, 220, 85, 170, 650] 
}

df = pd.DataFrame(data)
X = df.drop('price', axis=1)
y = df['price']

# We use a Pipeline to ensure scaling is handled automatically during prediction
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

pipeline.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("✅ Advanced Linear Regression Model Trained and Saved as model.pkl")