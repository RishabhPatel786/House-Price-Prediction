import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

# Training data with high-variance features to "teach" the linear model
# Features: sqft, bhk, age, tier, dist_road, dist_metro, furnish, floor, type, road_width
data = {
    'sqft': [800, 1200, 2500, 800, 1200, 2500, 5000, 1200, 1500, 3000],
    'bhk': [2, 3, 4, 2, 3, 4, 5, 2, 3, 4],
    'age': [10, 5, 2, 15, 5, 0, 1, 8, 3, 5],
    'tier': [0, 0, 1, 1, 2, 2, 2, 1, 2, 1], 
    'dist_road': [2.0, 0.5, 0.1, 5.0, 0.2, 0.05, 0.0, 1.0, 0.2, 0.5],
    'dist_metro': [10, 5, 2, 15, 1, 0.5, 0.2, 8, 2, 3],
    'furnish': [0, 1, 2, 0, 1, 2, 2, 1, 1, 2],
    'floor': [1, 5, 12, 1, 3, 25, 40, 2, 15, 8],
    'type': [0, 0, 1, 0, 0, 2, 2, 0, 1, 1],
    'road_width': [20, 30, 60, 15, 40, 100, 150, 30, 40, 60],
    'price': [12, 28, 75, 18, 55, 180, 1200, 38, 95, 145] 
}

df = pd.DataFrame(data)
X = df.drop('price', axis=1)
y = df['price']

# Linear Regression Pipeline
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LinearRegression())
])

model_pipeline.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model_pipeline, f)

print("✅ Linear Regression Brain trained with Multi-Feature Scaling.")