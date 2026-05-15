import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import numpy as np

# 2026 Calibrated Data
# tier: 0 (Village), 1 (Jabalpur/Bhopal), 2 (Indore/Ahmedabad)
data = {
    'sqft': [1000, 1000, 1000, 1500, 1500, 1500, 2000, 2000, 2000, 800, 800, 3000],
    'beds': [2, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 4],
    'baths': [2, 2, 2, 2, 2, 2, 3, 3, 3, 1, 1, 4],
    'tier': [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 2, 2], 
    'prop_type': [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 2], 
    'furnish': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    # Prices: Rural (25L) -> Standard (48L) -> Prime (78L)
    'price': [25, 48, 78, 38, 68, 115, 58, 98, 165, 18, 52, 360] 
}

df = pd.DataFrame(data)
X = df.drop('price', axis=1)
y = df['price']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("🚀 BRAIN UPDATED: Ready with 6 features.")