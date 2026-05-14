from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import requests
import os

app = Flask(__name__)

# Load the trained Linear Regression model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# City lists for Tier Detection
PRIME_CITIES = ['mumbai', 'delhi', 'bangalore', 'bengaluru', 'hyderabad', 'chennai', 'kolkata', 'pune', 'gurgaon', 'noida']
TIER2_CITIES = ['jabalpur', 'bhopal', 'gwalior', 'indore', 'jaipur', 'lucknow', 'nagpur', 'patna', 'ranchi']

@app.route('/')
def home():
    """Renders the main dashboard"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Parse Inputs from the Advanced Sidebar
        sqft      = float(request.form.get('sqft', 1))
        beds      = float(request.form.get('beds', 1))
        baths     = float(request.form.get('baths', 1))
        prop_type = int(request.form.get('prop_type', 0))
        furnish   = int(request.form.get('furnish', 0))
        lat       = request.form.get('lat', '')
        lng       = request.form.get('lng', '')

        # 2. BUG FIX: ILLOGICAL CONFIGURATION CHECK
        # Prevents impossible inputs like 10 bedrooms in a small flat.
        if sqft / beds < 200:
            return jsonify({
                'error': f'Architectural Error: {int(beds)} bedrooms in {int(sqft)} sq.ft. is not feasible.'
            }), 400

        # 3. Geo-Tier Detection via Nominatim
        tier = 0
        city_display = "Rural / Small Town"
        if lat and lng:
            try:
                headers = {'User-Agent': 'BharatEstateAI_Advanced_v2'}
                geo_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}"
                resp = requests.get(geo_url, headers=headers, timeout=5).json()
                addr = resp.get('display_name', '').lower()
                
                # Logic to determine city name for display
                addr_details = resp.get('address', {})
                city_display = addr_details.get('city') or addr_details.get('state_district') or "Detected Area"
                
                if any(c in addr for c in PRIME_CITIES):
                    tier = 2 # Metro
                elif any(c in addr for c in TIER2_CITIES):
                    tier = 1 # Tier-2
            except Exception:
                pass

        # 4. Model Prediction
        cols = ["sqft", "beds", "baths", "tier", "prop_type", "furnish"]
        features = pd.DataFrame([[sqft, beds, baths, tier, prop_type, furnish]], columns=cols)
        base_price = float(model.predict(features)[0])

        # 5. ADVANCED WEIGHTING (Fixes Rural vs City & Logic Bugs)
        # Apply Tier Multipliers (Manual adjustment to ensure distinct pricing)
        if tier == 0: base_price *= 0.60  # Rural/Village: 40% Cheaper
        if tier == 2: base_price *= 1.45  # Metro/Prime: 45% Premium
        
        # Property type impact (Manual weights for responsiveness)
        if prop_type == 1: base_price *= 1.20  # Independent House +20%
        if prop_type == 2: base_price *= 1.55  # Luxury Villa +55%
        
        # Furnishing impact
        if furnish == 1: base_price *= 1.12   # Semi-furnished +12%
        if furnish == 2: base_price *= 1.25   # Fully-furnished +25%

        # Formatting Function
        def fmt(val):
            if val >= 100:
                return f"₹{val/100:.2f} Cr"
            return f"₹{val:.1f} L"

        # 6. Final JSON Response
        return jsonify({
            'price': fmt(base_price),
            'price_low': fmt(base_price * 0.92),  # Lower confidence bound
            'price_high': fmt(base_price * 1.08), # Upper confidence bound
            'rate': f"₹{int((base_price * 100000) / sqft):,}/sqft",
            'city': city_display,
            'tier_label': ["Rural / Small Town", "Tier-2 City", "Metro / Prime City"][tier],
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': f'Prediction Engine Error: {str(e)}'}), 500

if __name__ == '__main__':
    # Running on local port 5000
    app.run(debug=True, port=5000)