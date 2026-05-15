from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import requests

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# 2026 Market Intelligence
PRIME = ['mumbai', 'delhi', 'bangalore', 'indore', 'pune', 'gurgaon', 'hyderabad']
STANDARD = ['jabalpur', 'bhopal', 'gwalior', 'jaipur', 'lucknow', 'nagpur', 'surat']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Gather Inputs
        sqft = float(request.form.get('sqft', 1200))
        beds = float(request.form.get('beds', 2))
        baths = float(request.form.get('baths', 2))
        prop_type = float(request.form.get('prop_type', 0))
        furnish = float(request.form.get('furnish', 0))
        lat, lng = request.form.get('lat'), request.form.get('lng')

        tier, multiplier, loc_tag, city = 0, 1.0, "Residential", "Unknown"

        # 2. Geospatial Intelligence
        if lat and lng:
            headers = {'User-Agent': 'BharatEstateAI_Pro_v9'}
            res = requests.get(f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}", headers=headers).json()
            full_addr = res.get('display_name', '').lower()
            city = res.get('address', {}).get('city') or res.get('address', {}).get('town') or "Area"

            if any(c in full_addr for c in PRIME):
                tier, multiplier, loc_tag = 2, 1.35, "High-Value Metro"
            elif any(c in full_addr for c in STANDARD):
                tier, multiplier, loc_tag = 1, 1.15, "Urban Growth Center"
            
            # Logic: Highway Proximity Boost
            if any(k in full_addr for k in ['highway', 'nh', 'bypass', 'expressway']):
                multiplier += 0.15
                loc_tag += " + Highway Access"

        # 3. Model Inference
        features = pd.DataFrame([[sqft, beds, baths, tier, prop_type, furnish]], 
                                columns=['sqft','beds','baths','tier','type','furnish'])
        prediction = model.predict(features)[0]
        
        final_price = max(prediction * multiplier, 10.0) # Market Floor

        # 4. JSON Response for AJAX
        return jsonify({
            'price': f"₹{final_price/100:.2f} Cr" if final_price >= 100 else f"₹{final_val:.1f} L" if 'final_val' in locals() else f"₹{final_price:.1f} L",
            'rate': f"₹{int((final_price * 100000) / sqft):,}/sqft",
            'city': city.title(),
            'loc_tag': loc_tag,
            'tier': tier
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500