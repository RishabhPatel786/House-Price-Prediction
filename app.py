from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import requests
import os

app = Flask(__name__)

# ============================================================
# Auto Create Model if Missing
# ============================================================

if not os.path.exists('model.pkl'):
    import model

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# ============================================================
# City Tier Mapping
# ============================================================

PRIME_CITIES = [
    'mumbai', 'delhi', 'bangalore', 'bengaluru',
    'hyderabad', 'pune', 'chennai',
    'kolkata', 'gurgaon', 'gurugram'
]

TIER2_CITIES = [
    'jabalpur', 'bhopal', 'indore',
    'raipur', 'nagpur', 'lucknow',
    'kanpur', 'patna'
]

# ============================================================
# Home Route
# ============================================================

@app.route('/')
def home():
    return render_template('index.html')

# ============================================================
# Predict Route
# ============================================================

@app.route('/predict', methods=['POST'])
def predict():

    try:

        sqft = float(request.form.get('sqft'))
        beds = float(request.form.get('beds'))
        baths = float(request.form.get('baths'))
        prop_type = float(request.form.get('prop_type'))
        furnish = float(request.form.get('furnish'))

        lat = request.form.get('lat')
        lng = request.form.get('lng')

        # ====================================================
        # Validation
        # ====================================================

        if sqft <= 0:
            return jsonify({
                'error': 'Area must be greater than 0'
            }), 400

        # ====================================================
        # Detect Tier
        # ====================================================

        tier = 0
        city_name = "Unknown Area"
        tier_label = "Rural Area"

        try:

            if lat and lng:

                url = (
                    f"https://nominatim.openstreetmap.org/reverse"
                    f"?format=json&lat={lat}&lon={lng}"
                )

                response = requests.get(
                    url,
                    headers={
                        "User-Agent":
                        "HousePricePredictionAI"
                    },
                    timeout=5
                )

                data = response.json()

                address = data.get('display_name', '').lower()

                city_name = (
                    data.get('address', {})
                    .get('city', 'Selected Area')
                )

                if any(
                    city in address
                    for city in PRIME_CITIES
                ):
                    tier = 2
                    tier_label = "Prime Metro"

                elif any(
                    city in address
                    for city in TIER2_CITIES
                ):
                    tier = 1
                    tier_label = "Tier-2 City"

                else:
                    tier = 0
                    tier_label = "Small Town"

        except:
            pass

        # ====================================================
        # Feature Engineering
        # ====================================================

        luxury_score = (
            (tier * 2) +
            (prop_type * 2) +
            furnish
        )

        room_density = (
            sqft /
            (beds + baths)
        )

        cols = [
            "sqft",
            "beds",
            "baths",
            "tier",
            "prop_type",
            "furnish",
            "luxury_score",
            "room_density"
        ]

        features = pd.DataFrame([[
            sqft,
            beds,
            baths,
            tier,
            prop_type,
            furnish,
            luxury_score,
            room_density
        ]], columns=cols)

        # ====================================================
        # Prediction
        # ====================================================

        base_price = float(
            model.predict(features)[0]
        )

        base_price = max(base_price, 3)
        base_price = min(base_price, 2500)

        low = base_price * 0.90
        high = base_price * 1.10

        # ====================================================
        # Market Labels
        # ====================================================

        if base_price < 25:
            market_status = "Affordable Housing"

        elif base_price < 80:
            market_status = "Mid-Range Property"

        elif base_price < 200:
            market_status = "Premium Property"

        else:
            market_status = "Luxury Real Estate"

        # ====================================================
        # Format Price
        # ====================================================

        def fmt(val):

            if val >= 100:
                return f"₹{val/100:.2f} Cr"

            return f"₹{val:.1f} L"

        rate_per_sqft = int(
            (base_price * 100000) / sqft
        )

        return jsonify({

            'price': fmt(base_price),
            'price_low': fmt(low),
            'price_high': fmt(high),

            'numeric_price': round(base_price, 2),

            'rate': f"₹{rate_per_sqft:,}/sqft",

            'market_status': market_status,

            'tier': tier,
            'tier_label': tier_label,

            'city': city_name.title(),

            'sqft': int(sqft),
            'beds': int(beds),
            'baths': int(baths)

        })

    except Exception as e:

        return jsonify({
            'error': str(e)
        }), 500

# ============================================================
# Run
# ============================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))

    app.run(
        host='0.0.0.0',
        port=port
    )
    