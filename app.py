from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import requests
import os

app = Flask(__name__)

# ─── 1. LOAD THE LINEAR REGRESSION PIPELINE ───
# Note: Ensure you have run 'python model.py' to generate the .pkl file
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("⚠️ Error: model.pkl not found. Please run model.py to train the model first.")

# ─── 2. MARKET DATA FOR TIER DETECTION (2026) ───
METROS = ['mumbai', 'delhi', 'bangalore', 'pune', 'indore', 'hyderabad', 'gurgaon', 'noida']
TIER2 = ['jabalpur', 'bhopal', 'gwalior', 'jaipur', 'lucknow', 'nagpur', 'surat', 'vadodara']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ─── 3. CAPTURE INPUTS FROM THE FORM ───
        # Numerical Inputs
        sqft = float(request.form.get('sqft', 1000))
        bhk = int(request.form.get('bhk', 2))
        age = int(request.form.get('age', 5))
        floor = int(request.form.get('floor', 1))
        road_width = int(request.form.get('road_width', 30))
        
        # Accessibility Distances (KM)
        d_road = float(request.form.get('dist_road', 0.5))
        d_metro = float(request.form.get('dist_metro', 5.0))
        
        # Categorical Inputs
        prop_type = int(request.form.get('type', 0))      # 0:Apt, 1:House, 2:Villa
        furnish = int(request.form.get('furnish', 1))     # 0:Un, 1:Semi, 2:Full
        
        lat = request.form.get('lat')
        lng = request.form.get('lng')

        # ─── 4. GEOSPATIAL INTELLIGENCE ───
        tier = 0
        multiplier = 1.0
        city_display = "Location Detected"

        if lat and lng and lat != "":
            # Using Nominatim API for reverse geocoding
            headers = {'User-Agent': 'BharatEstateAI_v12_Professional'}
            geo_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}"
            response = requests.get(geo_url, headers=headers, timeout=5).json()
            
            full_addr = response.get('display_name', '').lower()
            addr_details = response.get('address', {})
            city_display = addr_details.get('city') or addr_details.get('town') or addr_details.get('suburb') or "Area"

            # Determine Tier for the Linear Model
            if any(c in full_addr for c in METROS):
                tier = 2
                multiplier = 1.45  # 45% Premium for Metros
            elif any(c in full_addr for c in TIER2):
                tier = 1
                multiplier = 1.15  # 15% Premium for Tier-2

        # ─── 5. MODEL PREDICTION ───
        # Creating a DataFrame ensures feature names match the scaler in the pipeline
        features = pd.DataFrame([[sqft, bhk, age, tier, d_road, d_metro, furnish, floor, prop_type, road_width]], 
                                columns=['sqft','bhk','age','tier','dist_road','dist_metro','furnish','floor','type','road_width'])
        
        # Linear Regression Baseline
        base_prediction = model.predict(features)[0]

        # ─── 6. DYNAMIC MULTIPLIERS (Expert Logic) ───
        # Amenities Impact
        if request.form.get('amenities') == 'on': multiplier += 0.12
        if request.form.get('gated') == 'on': multiplier += 0.10
        if request.form.get('main_road_touch') == 'on': multiplier += 0.20
        
        # Distance Penalty (Based on Indian Property Rate Classification)
        if d_road > 2.0: multiplier *= 0.65  # 35% discount for deep interior
        elif d_road > 0.5: multiplier *= 0.88 # 12% discount

        final_val = base_prediction * multiplier

        # Safety Guardrails (Ensuring realistic pricing)
        if tier == 2: final_val = max(final_val, 45.0) 
        elif tier == 1: final_val = max(final_val, 25.0)
        else: final_val = max(final_val, 8.5)

        # ─── 7. RETURN JSON RESPONSE ───
        return jsonify({
            'price': f"₹{final_val/100:.2f} Cr" if final_val >= 100 else f"₹{final_val:.1f} Lakh",
            'rate': f"₹{int((final_val * 100000) / sqft):,}/sqft",
            'city': city_display.title()
        })

    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        return jsonify({'error': 'Server error. Please check your inputs.'}), 500

if __name__ == '__main__':
    # Dynamic port for Render deployment
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)