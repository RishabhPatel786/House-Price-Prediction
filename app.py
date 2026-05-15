from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import requests
import os

app = Flask(__name__)

# ─── 1. LOAD THE LINEAR REGRESSION BRAIN ───
# Ensure you have run 'python model.py' to generate this file
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("⚠️ Error: model.pkl not found. Run model.py first!")

# ─── 2. MARKET INTELLIGENCE DATA (2026) ───
METROS = ['mumbai', 'delhi', 'bangalore', 'pune', 'indore', 'hyderabad', 'gurgaon', 'noida', 'chennai', 'kolkata']
TIER2_CITIES = ['jabalpur', 'bhopal', 'gwalior', 'jaipur', 'lucknow', 'nagpur', 'surat', 'vadodara', 'kanpur', 'patna']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ─── 3. CAPTURE INPUTS FROM FRONTEND ───
        sqft = float(request.form.get('sqft', 1000))
        bhk = int(request.form.get('bhk', 2))
        age = int(request.form.get('age', 5))
        floor = int(request.form.get('floor', 1))
        road_width = int(request.form.get('road_width', 30))
        
        # Accessibility Distances (KM)
        d_road = float(request.form.get('dist_road', 0.5))
        d_metro = float(request.form.get('dist_metro', 5.0))
        
        # Categorical Selection
        prop_type = int(request.form.get('type', 0))      # 0: Apt, 1: House, 2: Villa
        furnish = int(request.form.get('furnish', 1))     # 0: Un, 1: Semi, 2: Full
        
        lat = request.form.get('lat')
        lng = request.form.get('lng')

        # ─── 4. GEOSPATIAL TIER DETECTION ───
        tier = 0
        multiplier = 1.0
        city_name = "Selected Locality"

        if lat and lng and lat != "":
            headers = {'User-Agent': 'BharatEstateAI_Production_v12'}
            geo_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}"
            res = requests.get(geo_url, headers=headers, timeout=5).json()
            
            full_addr = res.get('display_name', '').lower()
            addr_data = res.get('address', {})
            city_name = addr_data.get('city') or addr_data.get('town') or addr_data.get('suburb') or "Area"

            # Assign Tier Weight (Impacts the Linear Regression math)
            if any(c in full_addr for c in METROS):
                tier = 2
                multiplier = 1.45  # 45% Base Premium for Metros
            elif any(c in full_addr for c in TIER2_CITIES):
                tier = 1
                multiplier = 1.15  # 15% Base Premium for Tier-2

        # ─── 5. PREPARATION FOR LINEAR REGRESSION ───
        # Column order MUST match the model.py training script
        input_data = pd.DataFrame([[sqft, bhk, age, tier, d_road, d_metro, furnish, floor, prop_type, road_width]], 
                                columns=['sqft','bhk','age','tier','dist_road','dist_metro','furnish','floor','type','road_width'])
        
        # Calculate Base Price (Lakhs)
        prediction = model.predict(input_data)[0]

        # ─── 6. DYNAMIC EXPERT ADJUSTMENTS (DISTANCE & AMENITIES) ───
        # Amenities Multipliers
        if request.form.get('amenities') == 'on': multiplier += 0.12  # Pool/Gym
        if request.form.get('gated') == 'on': multiplier += 0.08     # Security
        if request.form.get('main_road_touch') == 'on': multiplier += 0.20 # Road Touch
        if request.form.get('lift') == 'on': multiplier += 0.05      # Convenience

        # Road Distance Penalties (Based on your provided chart)
        if d_road > 5.0: multiplier *= 0.40    # 60% Drop for interior village
        elif d_road > 2.0: multiplier *= 0.60  # 40% Drop
        elif d_road > 0.5: multiplier *= 0.85  # 15% Drop
        
        # Apply Multiplier to AI Prediction
        final_val = prediction * multiplier

        # ─── 7. SAFETY GUARDRAILS (PRICE FLOORS) ───
        if tier == 2: final_val = max(final_val, 45.0) # Metro property min 45L
        elif tier == 1: final_val = max(final_val, 25.0) # City property min 25L
        else: final_val = max(final_val, 8.5) # Village property min 8.5L

        # ─── 8. RETURN JSON TO FRONTEND ───
        return jsonify({
            'price': f"₹{final_val/100:.2f} Cr" if final_val >= 100 else f"₹{final_val:.1f} Lakh",
            'rate': f"₹{int((final_val * 100000) / sqft):,}/sqft",
            'city': city_name.title()
        })

    except Exception as e:
        print(f"🔥 Backend Error: {str(e)}")
        return jsonify({'error': 'Prediction failed. Check input data.'}), 500

if __name__ == '__main__':
    # Use environment port for Render deployment
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)