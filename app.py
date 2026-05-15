from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import requests
import os

app = Flask(__name__)

# Load stratified LR models (one per tier)
with open('model.pkl', 'rb') as f:
    models = pickle.load(f)

FEATURES = ['sqft', 'beds', 'baths', 'prop_type', 'furnish']

PRIME_CITIES = [
    'mumbai', 'delhi', 'bangalore', 'bengaluru', 'hyderabad', 'chennai',
    'kolkata', 'pune', 'ahmedabad', 'indore', 'surat', 'new delhi',
    'navi mumbai', 'thane', 'gurgaon', 'gurugram', 'noida', 'faridabad',
    'ghaziabad', 'kalyan', 'vasai', 'visakhapatnam', 'vijayawada'
]

TIER2_CITIES = [
    'jabalpur', 'bhopal', 'gwalior', 'vadodara', 'rajkot', 'nagpur',
    'nashik', 'aurangabad', 'coimbatore', 'lucknow', 'kanpur', 'jaipur',
    'patna', 'bhubaneswar', 'chandigarh', 'dehradun', 'mysuru', 'mysore',
    'mangalore', 'hubli', 'dharwad', 'amritsar', 'ludhiana', 'agra',
    'varanasi', 'meerut', 'raipur', 'ranchi', 'jodhpur', 'udaipur',
    'kota', 'ajmer', 'sagar', 'ujjain', 'rewa', 'satna', 'katni',
    'chhindwara', 'betul', 'sehore', 'hoshangabad', 'itarsi', 'damoh',
    'tiruchirappalli', 'madurai', 'salem', 'tirupur', 'erode', 'vellore',
    'warangal', 'karimnagar', 'nizamabad', 'kochi', 'thiruvananthapuram',
    'kozhikode', 'thrissur', 'kollam', 'guntur', 'nellore', 'kurnool'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/report')
def report():
    return render_template('report.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        sqft      = float(request.form.get('sqft', 1))
        beds      = float(request.form.get('beds', 0))
        baths     = float(request.form.get('baths', 0))
        prop_type = float(request.form.get('prop_type', 0))
        furnish   = float(request.form.get('furnish', 0))
        amenities = bool(request.form.get('amenities'))
        lat       = request.form.get('lat', '').strip()
        lng       = request.form.get('lng', '').strip()

        if sqft < 100:
            return jsonify({'error': 'Area must be at least 100 sq ft'}), 400
        if beds < 1 or baths < 1:
            return jsonify({'error': 'Beds and baths must be at least 1'}), 400

        # Validate beds vs sqft (realistic check)
        max_beds = max(1, int(sqft // 200))
        if beds > max_beds:
            return jsonify({'error': f'Too many bedrooms for {int(sqft)} sqft. Max realistic: {max_beds} beds.'}), 400

        # ── Geo tier detection ────────────────────────────────────
        tier = 0
        location_label = "Rural / Village"
        city_display = "Unknown Area"

        if lat and lng:
            try:
                headers = {'User-Agent': 'HousePricePredictorProject_v5_Educational'}
                geo_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}&addressdetails=1"
                resp = requests.get(geo_url, headers=headers, timeout=6)
                resp.raise_for_status()
                geo_data = resp.json()
                addr = geo_data.get('address', {})
                full_address = geo_data.get('display_name', '').lower()
                city_display = (
                    addr.get('city') or addr.get('town') or
                    addr.get('suburb') or addr.get('county') or
                    addr.get('state_district') or "Selected Area"
                )
                if any(c in full_address for c in PRIME_CITIES):
                    tier = 2; location_label = "Prime Metro City"
                elif any(c in full_address for c in TIER2_CITIES):
                    tier = 1; location_label = "Tier-2 City"
                else:
                    tier = 0; location_label = "Rural / Small Town"
            except requests.exceptions.Timeout:
                city_display = "Location (timeout)"
            except Exception:
                city_display = "Location Detected"

        # ── Predict using the correct tier's LR model ─────────────
        row = pd.DataFrame([[sqft, beds, baths, prop_type, furnish]], columns=FEATURES)
        base_price = float(models[tier].predict(row)[0])
        base_price = max(base_price, 1.5)

        if amenities:
            base_price *= 1.15

        low  = base_price * 0.88
        high = base_price * 1.12

        def fmt(val):
            if val >= 100: return f"₹{val/100:.2f} Cr"
            return f"₹{val:.1f} L"

        rate_per_sqft = int((base_price * 100000) / sqft) if sqft > 0 else 0

        prop_labels    = {0: "Apartment", 1: "Independent House", 2: "Luxury Villa"}
        furnish_labels = {0: "Unfurnished", 1: "Semi-Furnished", 2: "Fully Furnished"}

        return jsonify({
            'price':      fmt(base_price),
            'price_low':  fmt(low),
            'price_high': fmt(high),
            'rate':       f"₹{rate_per_sqft:,}/sqft",
            'tier':       tier,
            'tier_label': location_label,
            'city':       city_display.title(),
            'prop_type':  prop_labels.get(int(prop_type), "Property"),
            'furnish':    furnish_labels.get(int(furnish), ""),
            'sqft':       int(sqft),
            'beds':       int(beds),
            'baths':      int(baths),
            'amenities':  amenities,
        })

    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)