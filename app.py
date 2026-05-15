from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import requests

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Inputs from Form
        sqft = float(request.form.get('sqft', 1000))
        dist_cat = int(request.form.get('distance', 0)) # User input: 0 to 4
        lat, lng = request.form.get('lat'), request.form.get('lng')

        tier, multiplier, city = 0, 1.0, "Area"

        # Geo-Detection
        if lat and lng:
            headers = {'User-Agent': 'BharatEstateAI_v11'}
            res = requests.get(f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}", headers=headers).json()
            addr = res.get('display_name', '').lower()
            city = res.get('address', {}).get('city') or res.get('address', {}).get('town') or "Location"

            if any(c in addr for c in ['mumbai', 'delhi', 'bangalore', 'pune', 'indore']): tier = 2
            elif any(c in addr for c in ['jabalpur', 'bhopal', 'gwalior', 'jaipur']): tier = 1

        # Distance Multiplier Logic (Based on your data)
        # 0: Road touch, 1: <500m, 2: 500m-1km, 3: 1-2km, 4: 2km+
        dist_multipliers = {0: 1.0, 1: 0.85, 2: 0.75, 3: 0.60, 4: 0.40}
        penalty = dist_multipliers.get(dist_cat, 0.5)

        # AI Prediction
        features = pd.DataFrame([[sqft, tier, dist_cat, 0]], columns=['sqft','tier','dist','type'])
        base_val = model.predict(features)[0]
        
        final_val = base_val * penalty
        
        # Hard Floors for Cities
        if tier == 2: final_val = max(final_val, 45.0)
        elif tier == 1: final_val = max(final_val, 25.0)

        return jsonify({
            'price': f"₹{final_val/100:.2f} Cr" if final_val >= 100 else f"₹{final_val:.1f} L",
            'rate': f"₹{int((final_val * 100000) / sqft):,}/sqft",
            'city': city.title()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500