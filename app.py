from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import requests
import os

app = Flask(__name__)

# Load the trained Linear Regression pipeline
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Intelligence Keywords
HIGHWAY_KEYWORDS = ['highway', 'nh', 'bypass', 'expressway', 'road', 'marg', 'cloverleaf']
VILLAGE_KEYWORDS = ['village', 'gram', 'rural', 'kheda', 'basti', 'panchayat']
PRIME_CITIES = ['mumbai', 'delhi', 'bangalore', 'bengaluru', 'hyderabad', 'pune', 'indore', 'gurgaon', 'noida']
TIER2_CITIES = ['jabalpur', 'bhopal', 'gwalior', 'vadodara', 'jaipur', 'lucknow', 'kanpur', 'nagpur', 'surat']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ── 1. Parse Inputs ──
        sqft = float(request.form.get('sqft', 1))
        beds = float(request.form.get('beds', 1))
        baths = float(request.form.get('baths', 1))
        prop_type = float(request.form.get('prop_type', 0))
        furnish = float(request.form.get('furnish', 0))
        amenities = request.form.get('amenities') == 'true'
        lat, lng = request.form.get('lat'), request.form.get('lng')

        # ── 2. Intelligent Geo-Detection ──
        tier = 0
        location_label = "Standard Residential"
        city_display = "Unknown Location"
        context_multiplier = 1.0

        if lat and lng:
            headers = {'User-Agent': 'BharatEstate_AI_Professional_V2'}
            geo_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}&addressdetails=1"
            resp = requests.get(geo_url, headers=headers, timeout=5).json()
            
            addr = resp.get('address', {})
            full_address = resp.get('display_name', '').lower()
            city_display = addr.get('city') or addr.get('town') or addr.get('village') or addr.get('suburb') or "Selected Area"

            # A. Detect Highway Proximity (20% Premium)
            if any(key in full_address for key in HIGHWAY_KEYWORDS):
                context_multiplier += 0.20
                location_label = "Prime Connectivity Zone (Highway/Main Road)"
            
            # B. Assign Tier based on City lists
            if any(c in full_address for c in PRIME_CITIES):
                tier = 2
                location_label = "Metro / Prime Zone" if tier == 2 else location_label
            elif any(c in full_address for c in TIER2_CITIES):
                tier = 1
                location_label = "Developed Urban Center"
            
            # C. Detect Village/Rural Context (15% Discount)
            if any(key in full_address for key in VILLAGE_KEYWORDS):
                tier = 0
                context_multiplier -= 0.15
                location_label = "Rural / Village Settlement"

        # ── 3. Model Prediction ──
        features = pd.DataFrame([[sqft, beds, baths, tier, prop_type, furnish]], 
                                columns=["sqft","beds","baths","tier","prop_type","furnish"])
        
        # Linear Regression Baseline
        prediction = float(model.predict(features)[0])
        
        # Apply Geospatial Context
        final_price = prediction * context_multiplier
        if amenities: final_price *= 1.12 # 12% boost for amenities
        
        # Logical price floor by Tier
        floors = {0: 4.5, 1: 12.0, 2: 30.0}
        final_price = max(final_price, floors.get(tier, 5.0))

        # ── 4. Format Result ──
        def fmt(v): return f"₹{v/100:.2f} Cr" if v >= 100 else f"₹{v:.1f} L"
        
        return jsonify({
            'price': fmt(final_price),
            'rate': f"₹{int((final_price * 100000) / sqft):,}/sqft",
            'tier_label': location_label,
            'city': city_display.title(),
            'status': 'Success'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)