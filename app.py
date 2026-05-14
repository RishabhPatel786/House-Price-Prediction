from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import requests

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

PRIME_CITIES = ['mumbai', 'delhi', 'bangalore', 'bengaluru', 'hyderabad', 'chennai', 'kolkata', 'pune', 'gurgaon', 'noida']
TIER2_CITIES = ['jabalpur', 'bhopal', 'gwalior', 'indore', 'jaipur', 'lucknow', 'nagpur', 'patna', 'ranchi']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        sqft = float(request.form.get('sqft', 1))
        beds = float(request.form.get('beds', 1))
        baths = float(request.form.get('baths', 1))
        prop_type = int(request.form.get('prop_type', 0))
        furnish = int(request.form.get('furnish', 0))
        lat = request.form.get('lat', '')
        lng = request.form.get('lng', '')
        # FIX: Capture the exact name from the search bar
        manual_city = request.form.get('manual_city', '')

        if sqft / beds < 200:
            return jsonify({'error': f'Architectural Error: {int(beds)} beds in {int(sqft)} sqft is not possible.'}), 400

        tier = 0
        city_display = manual_city or "Rural Area"

        # Even if we have a manual city name, we still need the tier for the model
        if lat and lng:
            try:
                headers = {'User-Agent': 'BharatEstateAI_v5'}
                geo_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}&countrycodes=in&addressdetails=1"
                resp = requests.get(geo_url, headers=headers, timeout=5).json()
                
                if resp.get('address', {}).get('country_code') != 'in':
                    return jsonify({'error': 'Location Error: Please select a location within India.'}), 400

                addr_full = resp.get('display_name', '').lower()
                
                # If no manual city was passed (user clicked map instead of searching), detect it
                if not manual_city:
                    addr_details = resp.get('address', {})
                    locality = addr_details.get('suburb') or addr_details.get('neighbourhood') or addr_details.get('village')
                    city = addr_details.get('city') or addr_details.get('town') or addr_details.get('state_district')
                    city_display = f"{locality}, {city}" if locality and city else (locality or city or "Detected Area")

                if any(c in addr_full for c in PRIME_CITIES): tier = 2
                elif any(c in addr_full for c in TIER2_CITIES): tier = 1
            except: pass

        # Prediction & Multipliers
        features = pd.DataFrame([[sqft, beds, baths, tier, prop_type, furnish]], columns=["sqft","beds","baths","tier","prop_type","furnish"])
        base_price = float(model.predict(features)[0])

        if tier == 0: base_price *= 0.60
        if tier == 2: base_price *= 1.45
        if prop_type == 1: base_price *= 1.20
        if prop_type == 2: base_price *= 1.55
        if furnish == 1: base_price *= 1.12
        if furnish == 2: base_price *= 1.25

        def fmt(val): return f"₹{val/100:.2f} Cr" if val >= 100 else f"₹{val:.1f} L"

        return jsonify({
            'price': fmt(base_price),
            'price_low': fmt(base_price * 0.92),
            'price_high': fmt(base_price * 1.08),
            'city': city_display,
            'tier_label': ["Rural Area", "Tier-2 City", "Metro City"][tier]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)