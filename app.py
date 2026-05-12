from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import requests
import os

app = Flask(__name__)

# Load Model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Intelligence Arrays
HIGHWAY_KEYS = ['highway', 'nh', 'bypass', 'expressway', 'road', 'marg', 'cloverleaf']
VILLAGE_KEYS = ['village', 'gram', 'rural', 'kheda', 'basti', 'panchayat']
PRIME_CITIES = ['mumbai', 'delhi', 'bangalore', 'hyderabad', 'pune', 'indore', 'gurgaon', 'noida']
TIER2_CITIES = ['jabalpur', 'bhopal', 'gwalior', 'jaipur', 'lucknow', 'nagpur', 'surat']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        sqft = float(data.get('sqft', 1000))
        beds = float(data.get('beds', 2))
        baths = float(data.get('baths', 2))
        prop_type = float(data.get('prop_type', 0))
        furnish = float(data.get('furnish', 1))
        lat, lng = data.get('lat'), data.get('lng')

        tier, multiplier, loc_label, city_name = 0, 1.0, "Residential Zone", "Unknown"

        if lat and lng:
            headers = {'User-Agent': 'BharatEstate_AI_V2'}
            res = requests.get(f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}", headers=headers).json()
            addr = res.get('display_name', '').lower()
            addr_details = res.get('address', {})
            city_name = addr_details.get('city') or addr_details.get('town') or addr_details.get('village') or "Point"

            if any(k in addr for k in HIGHWAY_KEYS):
                multiplier += 0.22
                loc_label = "Premium Connectivity (Near Highway)"
            
            if any(c in addr for c in PRIME_CITIES): tier = 2
            elif any(c in addr for c in TIER2_CITIES): tier = 1
            
            if any(k in addr for k in VILLAGE_KEYS):
                tier = 0
                multiplier -= 0.15
                loc_label = "Rural / Village Area"

        # Statistical Inference
        features = pd.DataFrame([[sqft, beds, baths, tier, prop_type, furnish]], 
                                columns=["sqft","beds","baths","tier","prop_type","furnish"])
        
        base_val = float(model.predict(features)[0])
        final_val = max(base_val * multiplier, 4.5)

        return jsonify({
            'price': f"₹{final_val/100:.2f} Cr" if final_val >= 100 else f"₹{final_val:.1f} L",
            'rate': f"₹{int((final_val * 100000) / sqft):,}/sqft",
            'label': loc_label,
            'city': city_name.title()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)