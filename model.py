import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

# Calibrated Dataset with Distance Impact
# dist: 0 (On Road), 1 (0-500m), 2 (500m-2km), 3 (2-5km), 4 (5km+)
data = {
    'sqft':   [1000, 1000, 1000, 1000, 1000, 1500, 1500, 1500, 2000, 2000],
    'tier':   [0,    0,    1,    1,    2,    0,    1,    2,    1,    2],
    'dist':   [3,    0,    2,    0,    0,    4,    1,    0,    2,    0],
    'type':   [0,    0,    0,    0,    0,    1,    1,    1,    1,    2],
    'price':  [12,   35,   45,   95,   210,  18,   85,   280,  110,  750] 
}

df = pd.DataFrame(data)
X = df.drop('price', axis=1)
y = df['price']

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LinearRegression())
])

pipeline.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("✅ BRAIN UPDATED: Distance-based logic enabled.")