from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import requests
import os
import json

app = Flask(__name__)

# Load the trained Linear Regression model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# City tier mapping
PRIME_CITIES = [
    'mumbai', 'delhi', 'bangalore', 'bengaluru', 'hyderabad',
    'chennai', 'kolkata', 'pune', 'ahmedabad', 'indore', 'surat',
    'new delhi', 'navi mumbai', 'thane', 'gurgaon', 'gurugram', 'noida'
]

TIER2_CITIES = [
    'jabalpur', 'bhopal', 'gwalior', 'vadodara', 'rajkot',
    'nagpur', 'nashik', 'aurangabad', 'coimbatore', 'lucknow',
    'kanpur', 'jaipur', 'patna', 'bhubaneswar', 'chandigarh',
    'dehradun', 'mysuru', 'mysore', 'mangalore', 'hubli', 'dharwad',
    'amritsar', 'ludhiana', 'agra', 'varanasi', 'meerut', 'raipur',
    'ranchi', 'jodhpur', 'udaipur', 'kota', 'ajmer', 'sagar',
    'ujjain', 'rewa', 'satna', 'katni', 'chhindwara', 'betul'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ── 1. Parse Inputs ──────────────────────────────────────────
        sqft      = float(request.form.get('sqft', 1))
        beds      = float(request.form.get('beds', 0))
        baths     = float(request.form.get('baths', 0))
        prop_type = float(request.form.get('prop_type', 0))
        furnish   = float(request.form.get('furnish', 0))
        amenities = bool(request.form.get('amenities'))
        lat       = request.form.get('lat', '').strip()
        lng       = request.form.get('lng', '').strip()

        # Input sanity checks
        if sqft <= 0:
            return jsonify({'error': 'Area must be greater than 0'}), 400
        if beds < 1 or baths < 1:
            return jsonify({'error': 'Beds and baths must be at least 1'}), 400

        # ── 2. Geo-based Tier Detection ──────────────────────────────
        tier = 0
        location_label = "Rural / Village"
        city_display = "Unknown Area"

        if lat and lng:
            try:
                headers = {'User-Agent': 'HousePricePredictorProject_v4_Educational'}
                geo_url = (
                    f"https://nominatim.openstreetmap.org/reverse"
                    f"?format=json&lat={lat}&lon={lng}&addressdetails=1"
                )
                resp = requests.get(geo_url, headers=headers, timeout=6)
                resp.raise_for_status()
                geo_data = resp.json()

                addr = geo_data.get('address', {})
                full_address = geo_data.get('display_name', '').lower()

                # Best city name fallback chain
                city_display = (
                    addr.get('city') or
                    addr.get('town') or
                    addr.get('suburb') or
                    addr.get('county') or
                    addr.get('state_district') or
                    "Selected Area"
                )

                if any(c in full_address for c in PRIME_CITIES):
                    tier = 2
                    location_label = "Prime Metro City"
                elif any(c in full_address for c in TIER2_CITIES):
                    tier = 1
                    location_label = "Tier-2 City"
                else:
                    tier = 0
                    location_label = "Rural / Small Town"

            except requests.exceptions.Timeout:
                city_display = "Location (timeout)"
            except Exception:
                city_display = "Location Detected"

        # ── 3. Predict via Linear Regression ─────────────────────────
        cols = ["sqft","beds","baths","tier","prop_type","furnish"]
        features = pd.DataFrame([[sqft, beds, baths, tier, prop_type, furnish]], columns=cols)
        base_price = float(model.predict(features)[0])
        base_price = max(base_price, 3.0)  # floor: ₹3 Lakh

        # Premium amenities uplift
        if amenities:
            base_price *= 1.15

        # Confidence interval (±12% for LR model)
        low  = base_price * 0.88
        high = base_price * 1.12

        # ── 4. Format Results ─────────────────────────────────────────
        def fmt(val):
            if val >= 100:
                return f"₹{val/100:.2f} Cr"
            return f"₹{val:.1f} L"

        rate_per_sqft = int((base_price * 100000) / sqft) if sqft > 0 else 0

        # Property type label
        prop_labels = {0: "Apartment", 1: "Independent House", 2: "Luxury Villa"}
        furnish_labels = {0: "Unfurnished", 1: "Semi-Furnished", 2: "Fully Furnished"}

        return jsonify({
            'price':        fmt(base_price),
            'price_low':    fmt(low),
            'price_high':   fmt(high),
            'rate':         f"₹{rate_per_sqft:,}/sqft",
            'tier':         tier,
            'tier_label':   location_label,
            'city':         city_display.title(),
            'prop_type':    prop_labels.get(int(prop_type), "Property"),
            'furnish':      furnish_labels.get(int(furnish), ""),
            'sqft':         int(sqft),
            'beds':         int(beds),
            'baths':        int(baths),
            'amenities':    amenities,
        })

    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)