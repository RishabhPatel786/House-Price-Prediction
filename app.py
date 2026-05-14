from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import requests
import os

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# City tier lists
PRIME_CITIES = ['mumbai', 'delhi', 'bangalore', 'bengaluru', 'hyderabad', 'chennai', 'kolkata', 'pune', 'gurgaon', 'noida']
TIER2_CITIES = ['jabalpur', 'bhopal', 'gwalior', 'indore', 'jaipur', 'lucknow', 'nagpur', 'patna', 'ranchi']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse Inputs
        sqft      = float(request.form.get('sqft', 1))
        beds      = float(request.form.get('beds', 1))
        baths     = float(request.form.get('baths', 1))
        prop_type = int(request.form.get('prop_type', 0))
        furnish   = int(request.form.get('furnish', 0))
        amenities = request.form.get('amenities') == 'on'
        lat       = request.form.get('lat', '')
        lng       = request.form.get('lng', '')

        # --- FIX 4: ILLOGICAL BEDROOM CHECK ---
        # A bedroom needs at least 200 sqft of space usually.
        if sqft / beds < 200:
            return jsonify({'error': f'Impossible Config: {int(beds)} bedrooms in {int(sqft)} sqft would be too small to exist.'}), 400

        # Geo-Tier Detection
        tier = 0
        city_display = "Rural / Small Town"
        if lat and lng:
            try:
                headers = {'User-Agent': 'Mozilla/5.0'}
                geo_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}"
                resp = requests.get(geo_url, headers=headers, timeout=5).json()
                addr = resp.get('display_name', '').lower()
                city_display = resp.get('address', {}).get('city') or resp.get('address', {}).get('state_district') or "Detected Area"
                
                if any(c in addr for c in PRIME_CITIES):
                    tier = 2
                elif any(c in addr for c in TIER2_CITIES):
                    tier = 1
            except:
                pass

        # Prediction
        cols = ["sqft","beds","baths","tier","prop_type","furnish"]
        features = pd.DataFrame([[sqft, beds, baths, tier, prop_type, furnish]], columns=cols)
        base_price = float(model.predict(features)[0])

        # --- FIX 4 & 5: MANUAL WEIGHTING FOR REALISM ---
        # Tier impact
        if tier == 0: base_price *= 0.65  # Rural is 35% cheaper
        if tier == 2: base_price *= 1.45  # Metro is 45% more expensive

        # Property type impact
        if prop_type == 1: base_price *= 1.20 # House premium
        if prop_type == 2: base_price *= 1.50 # Villa premium

        # Furnishing impact
        if furnish == 1: base_price *= 1.12 # Semi-furnished
        if furnish == 2: base_price *= 1.25 # Fully-furnished

        if amenities: base_price *= 1.15

        # Floor price
        base_price = max(base_price, 5.0) 

        def fmt(val):
            return f"₹{val/100:.2f} Cr" if val >= 100 else f"₹{val:.1f} L"

        return jsonify({
            'price': fmt(base_price),
            'price_low': fmt(base_price * 0.88),
            'price_high': fmt(base_price * 1.12),
            'rate': f"₹{int((base_price * 100000) / sqft):,}/sqft",
            'city': city_display,
            'tier_label': ["Rural Area", "Tier-2 City", "Metro City"][tier],
            'sqft': int(sqft), 'beds': int(beds), 'baths': int(baths)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)