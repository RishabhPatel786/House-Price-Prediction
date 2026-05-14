from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import requests
import os

app = Flask(__name__)

# Load the trained Linear Regression model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Indian Metro and Tier-2 City Clusters
PRIME_CITIES = ['mumbai', 'delhi', 'bangalore', 'bengaluru', 'hyderabad', 'chennai', 'kolkata', 'pune', 'gurgaon', 'noida']
TIER2_CITIES = ['jabalpur', 'bhopal', 'gwalior', 'indore', 'jaipur', 'lucknow', 'nagpur', 'patna', 'ranchi']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Parse Inputs
        sqft      = float(request.form.get('sqft', 1))
        beds      = float(request.form.get('beds', 1))
        baths     = float(request.form.get('baths', 1))
        prop_type = int(request.form.get('prop_type', 0))
        furnish   = int(request.form.get('furnish', 0))
        lat       = request.form.get('lat', '')
        lng       = request.form.get('lng', '')

        # 2. Logistical Sanity Check
        if sqft / beds < 200:
            return jsonify({
                'error': f'Architectural Error: {int(beds)} bedrooms in {int(sqft)} sq.ft. is not feasible.'
            }), 400

        # 3. India-Only Tier Detection
        tier = 0
        city_display = "Rural / Small Town"
        if lat and lng:
            try:
                headers = {'User-Agent': 'BharatEstateAI_Final_v3'}
                geo_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}&countrycodes=in"
                resp = requests.get(geo_url, headers=headers, timeout=5).json()
                
                # Enforce India-only results
                if resp.get('address', {}).get('country_code') != 'in':
                    return jsonify({'error': 'Location Error: Please select a location within India.'}), 400

                addr = resp.get('display_name', '').lower()
                city_display = resp.get('address', {}).get('city') or resp.get('address', {}).get('state_district') or "Detected Area"
                
                if any(c in addr for c in PRIME_CITIES):
                    tier = 2
                elif any(c in addr for c in TIER2_CITIES):
                    tier = 1
            except Exception: pass

        # 4. Model Prediction & Realism Multipliers
        cols = ["sqft", "beds", "baths", "tier", "prop_type", "furnish"]
        features = pd.DataFrame([[sqft, beds, baths, tier, prop_type, furnish]], columns=cols)
        base_price = float(model.predict(features)[0])

        # Manual weights for distinct Pricing
        if tier == 0: base_price *= 0.60  # Rural discount
        if tier == 2: base_price *= 1.45  # Metro premium
        if prop_type == 1: base_price *= 1.20 # House
        if prop_type == 2: base_price *= 1.55 # Villa
        if furnish == 1: base_price *= 1.12 # Semi
        if furnish == 2: base_price *= 1.25 # Full

        def fmt(val):
            return f"₹{val/100:.2f} Cr" if val >= 100 else f"₹{val:.1f} L"

        return jsonify({
            'price': fmt(base_price),
            'price_low': fmt(base_price * 0.92),
            'price_high': fmt(base_price * 1.08),
            'rate': f"₹{int((base_price * 100000) / sqft):,}/sqft",
            'city': city_display,
            'tier_label': ["Rural Area", "Tier-2 City", "Metro / Prime City"][tier]
        })

    except Exception as e:
        return jsonify({'error': f'Engine Error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)