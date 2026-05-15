from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import requests

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Market Intelligence Lists
METROS = ['mumbai', 'delhi', 'bangalore', 'pune', 'indore', 'hyderabad', 'gurgaon', 'noida']
TIER2 = ['jabalpur', 'bhopal', 'gwalior', 'jaipur', 'lucknow', 'nagpur', 'surat', 'vadodara']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        sqft = float(request.form.get('sqft', 1000))
        beds = float(request.form.get('beds', 2))
        baths = float(request.form.get('baths', 2))
        prop_type = float(request.form.get('prop_type', 0))
        furnish = float(request.form.get('furnish', 0))
        lat, lng = request.form.get('lat'), request.form.get('lng')

        tier, multiplier, loc_label, city = 0, 1.0, "Rural/Village", "Point Selected"

        if lat and lng:
            headers = {'User-Agent': 'BharatEstateAI_Pro_v10'}
            res = requests.get(f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}", headers=headers).json()
            full_addr = res.get('display_name', '').lower()
            city = res.get('address', {}).get('city') or res.get('address', {}).get('town') or "Area"

            # Intelligence logic to force price separation
            if any(c in full_addr for c in METROS):
                tier, multiplier, loc_label = 2, 1.85, "High-Value Metro"
            elif any(c in full_addr for c in TIER2):
                tier, multiplier, loc_label = 1, 1.25, "Urban City Zone"
            
            # Additional Highway Boost
            if any(k in full_addr for k in ['highway', 'nh', 'bypass', 'expressway']):
                multiplier += 0.20
                loc_label += " (Highway Connectivity)"

        # Prediction via Polynomial Pipeline
        features = pd.DataFrame([[sqft, beds, baths, tier, prop_type, furnish]], 
                                columns=['sqft','beds','baths','tier','type','furnish'])
        
        base_prediction = model.predict(features)[0]
        
        # Applying the Locality Intelligence Multiplier
        final_price = max(base_prediction * multiplier, 8.5)

        # Safety: Forced minimum for Metros (2026 inflation)
        if tier == 2: final_price = max(final_price, 45.0) 

        return jsonify({
            'price': f"₹{final_price/100:.2f} Cr" if final_price >= 100 else f"₹{final_price:.1f} L",
            'rate': f"₹{int((final_price * 100000) / sqft):,}/sqft",
            'city': city.title(),
            'loc_tag': loc_label
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)