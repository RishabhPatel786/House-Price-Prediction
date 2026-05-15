import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
import numpy as np

# ─────────────────────────────────────────────────────────────
#  Expanded & Calibrated Dataset (2024-25 Indian Market Prices)
#  Features: sqft, beds, baths, tier, prop_type, furnish
#  tier:      0=Rural/Village, 1=Tier-2 City, 2=Metro/Prime
#  prop_type: 0=Apartment, 1=Independent House, 2=Luxury Villa
#  furnish:   0=Unfurnished, 1=Semi-Furnished, 2=Fully Furnished
#  price:     in Lakhs (₹)
# ─────────────────────────────────────────────────────────────
data = {
    'sqft':      [600,  700,  800,  900, 1000, 1000, 1000, 1100, 1200, 1200,
                  1300, 1500, 1500, 1500, 1600, 1800, 2000, 2000, 2000, 2200,
                  2500, 3000, 3500, 4000, 800,  900,  1000, 1200, 1500, 2000,
                  600,  750,  850,  950, 1050, 1150, 1250, 1350, 1450, 1600,
                  1700, 1900, 2100, 2300, 2600, 2800, 3200, 3800, 700,  1100],

    'beds':      [1, 1, 2, 2, 2, 2, 2, 2, 2, 3,
                  3, 3, 3, 3, 3, 3, 3, 3, 4, 4,
                  4, 4, 5, 5, 1, 2, 2, 3, 3, 4,
                  1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                  3, 4, 4, 4, 4, 5, 5, 6, 1, 2],

    'baths':     [1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                  2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
                  4, 4, 4, 5, 1, 1, 2, 2, 3, 3,
                  1, 1, 1, 1, 2, 2, 2, 2, 2, 3,
                  3, 3, 3, 4, 4, 4, 5, 5, 1, 2],

    'tier':      [0, 0, 0, 0, 0, 1, 2, 1, 1, 2,
                  1, 0, 1, 2, 2, 1, 0, 1, 2, 2,
                  2, 2, 2, 2, 2, 2, 1, 1, 1, 1,
                  0, 0, 0, 1, 1, 2, 0, 1, 2, 1,
                  2, 1, 2, 2, 2, 2, 2, 2, 0, 0],

    'prop_type': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  1, 0, 1, 0, 1, 1, 1, 1, 1, 1,
                  1, 2, 2, 2, 0, 0, 0, 1, 1, 1,
                  0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
                  1, 1, 1, 2, 2, 2, 2, 2, 0, 0],

    'furnish':   [0, 0, 1, 0, 1, 1, 1, 1, 2, 2,
                  1, 0, 1, 2, 2, 1, 0, 1, 2, 2,
                  2, 2, 2, 2, 0, 1, 1, 1, 2, 2,
                  0, 1, 0, 1, 1, 2, 0, 1, 2, 1,
                  2, 1, 2, 2, 2, 2, 2, 2, 0, 1],

    # Prices in Lakhs (₹)
    'price':     [8,  10, 14, 16, 22, 42, 72, 48, 55, 85,
                  60, 28, 65, 110, 130, 80, 52, 92, 160, 175,
                  210, 340, 420, 550, 75, 90, 50, 70, 95, 140,
                  7,  12, 15, 40, 45, 78, 35, 62, 95, 85,
                  145, 100, 185, 280, 320, 400, 480, 600, 9, 18]
}

df = pd.DataFrame(data)
X = df.drop('price', axis=1)
y = df['price']

# Linear Regression with feature scaling (required for proper LR)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

pipeline.fit(X, y)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

# Print model stats
from sklearn.metrics import r2_score
y_pred = pipeline.predict(X)
r2 = r2_score(y, y_pred)
coef = pipeline.named_steps['model'].coef_
print(f"✅ Linear Regression Model Trained!")
print(f"   R² Score   : {r2:.4f}")
print(f"   Features   : sqft, beds, baths, tier, prop_type, furnish")
print(f"   Coefficients: {np.round(coef, 3)}")
print(f"   Dataset    : {len(df)} samples")
print(f"🚀 model.pkl saved successfully.")