from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import requests

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# City Classifications for 2026 accuracy
METROS = ['mumbai', 'delhi', 'bangalore', 'pune', 'indore', 'hyderabad', 'gurgaon', 'noida']
TIER2 = ['jabalpur', 'bhopal', 'gwalior', 'jaipur', 'lucknow', 'nagpur', 'surat']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get inputs
        sqft = float(request.form.get('sqft', 1000))
        dist_road = float(request.form.get('dist_road', 0.5))
        lat = request.form.get('lat')
        lng = request.form.get('lng')

        # 2. Tier Detection
        tier = 0
        multiplier = 1.0
        if lat and lng:
            res = requests.get(f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}", headers={'User-Agent':'BE_AI'}).json()
            addr = res.get('display_name', '').lower()
            if any(c in addr for c in METROS):
                tier, multiplier = 2, 1.65  # 65% extra for Metros
            elif any(c in addr for c in TIER2):
                tier, multiplier = 1, 1.25  # 25% extra for Tier-2

        # 3. Create Feature Array
        # Must match model.py columns exactly
        features = pd.DataFrame([[
            sqft, 
            int(request.form.get('bhk', 2)), 
            int(request.form.get('age', 5)), 
            tier, 
            dist_road, 
            float(request.form.get('dist_metro', 5.0)),
            int(request.form.get('furnish', 1)), 
            int(request.form.get('floor', 1)), 
            int(request.form.get('type', 0)), 
            int(request.form.get('road_width', 30))
        ]], columns=['sqft','bhk','age','tier','dist_road','dist_metro','furnish','floor','type','road_width'])

        # 4. Math Calculation
        base_prediction = model.predict(features)[0]
        
        # Apply the Multiplier + Road Distance Impact
        if dist_road > 2.0: multiplier *= 0.60 # 40% discount for interior
        
        final_price = base_prediction * multiplier

        # 5. Accuracy Guard (Prevents negative or unrealistic low prices)
        if tier == 2: final_price = max(final_price, 55.0) 
        elif tier == 1: final_price = max(final_val, 28.0) if 'final_val' in locals() else max(final_price, 28.0)

        return jsonify({
            'price': f"₹{final_price/100:.2f} Cr" if final_price >= 100 else f"₹{final_price:.1f} Lakh",
            'rate': f"₹{int((final_price * 100000) / sqft):,}/sqft"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500