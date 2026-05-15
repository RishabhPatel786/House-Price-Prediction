import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import pickle
import numpy as np

# ─────────────────────────────────────────────────────────────
# APPROACH: Train separate LR models per tier so each tier has
# its own regression line. This is still Linear Regression but
# stratified — a common real-world technique called Stratified LR.
# ─────────────────────────────────────────────────────────────

np.random.seed(42)

def make_price(sqft, prop_type, furnish, tier):
    base_rates = {0: 850, 1: 3400, 2: 7800}   # ₹/sqft by tier
    type_mult   = {0: 1.0, 1: 1.20, 2: 1.60}
    furnish_mult= {0: 1.0, 1: 1.13, 2: 1.28}
    rate = base_rates[tier] * type_mult[prop_type] * furnish_mult[furnish]
    price = (sqft * rate) / 100000
    return round(price * np.random.uniform(0.93, 1.07), 1)

def beds_for(sqft):
    if sqft <= 600: return 1
    if sqft <= 950: return 2
    if sqft <= 1400: return 3
    if sqft <= 2000: return 4
    return 5

configs = []

# Rural
for sqft in [400,500,600,700,750,800,850,900,950,1000,1100,1200,1350,1500,1600,1800,2000,2200,2500]:
    for pt in [0,1]:
        for fn in [0,1,2]:
            b=beds_for(sqft)
            configs.append([sqft,b,max(1,b-1),0,pt,fn, make_price(sqft,pt,fn,0)])

# Tier-2
for sqft in [600,700,800,900,1000,1100,1200,1350,1500,1700,1800,2000,2200,2500,3000,3500]:
    for pt in [0,1,2]:
        for fn in [0,1,2]:
            b=beds_for(sqft)
            configs.append([sqft,b,max(1,b-1),1,pt,fn, make_price(sqft,pt,fn,1)])

# Metro
for sqft in [500,600,700,800,900,1000,1200,1400,1500,1800,2000,2500,3000,3500,4000,5000]:
    for pt in [0,1,2]:
        for fn in [0,1,2]:
            b=beds_for(sqft)
            configs.append([sqft,b,max(1,b-1),2,pt,fn, make_price(sqft,pt,fn,2)])

df = pd.DataFrame(configs, columns=['sqft','beds','baths','tier','prop_type','furnish','price'])
print(f"Dataset: {len(df)} samples")

# ── Train one Pipeline per tier ───────────────────────────────
models = {}
features = ['sqft','beds','baths','prop_type','furnish']

for tier in [0, 1, 2]:
    sub = df[df['tier'] == tier]
    X_sub = sub[features]
    y_sub = sub['price']
    pipe = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
    pipe.fit(X_sub, y_sub)
    y_pred = pipe.predict(X_sub)
    r2 = r2_score(y_sub, y_pred)
    coef = dict(zip(features, np.round(pipe.named_steps['lr'].coef_, 2)))
    print(f"  Tier {tier} R²={r2:.3f} | coef={coef}")
    models[tier] = pipe

# Save all 3 models
with open('model.pkl','wb') as f:
    pickle.dump(models, f)

# ── Sanity checks ──────────────────────────────────────────────
print("\nSanity checks:")
tests = [
    (800,2,1,0,0,0,"800sqft Rural   Unfurnished Apt"),
    (800,2,1,1,0,0,"800sqft Tier-2  Unfurnished Apt"),
    (800,2,1,2,0,0,"800sqft Metro   Unfurnished Apt"),
    (800,2,1,1,0,1,"800sqft Tier-2  Semi-Furnished Apt"),
    (800,2,1,1,0,2,"800sqft Tier-2  Fully-Furnished Apt"),
    (800,2,1,1,1,1,"800sqft Tier-2  Semi-Furnished House"),
    (800,2,1,1,2,2,"800sqft Tier-2  Fully-Furnished Villa"),
    (1500,3,2,1,0,1,"1500sqft Tier-2 Semi Apt"),
    (2000,4,3,2,1,2,"2000sqft Metro  Fully House"),
]
for sqft,beds,baths,tier,pt,fn,label in tests:
    row = pd.DataFrame([[sqft,beds,baths,pt,fn]], columns=features)
    p = max(1.0, models[tier].predict(row)[0])
    print(f"  {label:45s} → ₹{p:.1f}L")

print("\n✅ Stratified Linear Regression models saved to model.pkl")