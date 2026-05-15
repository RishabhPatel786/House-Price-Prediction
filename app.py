from flask import Flask, render_template, request
import pickle
import numpy as np
import requests
import os

app = Flask(__name__)

# Load the brain
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Capture Inputs
        sqft = float(request.form.get('sqft', 1))
        beds = float(request.form.get('beds', 0))
        baths = float(request.form.get('baths', 0))
        prop_type = float(request.form.get('prop_type', 0))
        furnish = float(request.form.get('furnish', 0))
        lat = request.form.get('lat')
        lng = request.form.get('lng')

        # 2. Strict City & Tier Detection
        tier = 0 
        location_label = "Rural/Village"
        city_display = "Location Detected"

        if lat and lng and lat != "":
            try:
                # Mandatory User-Agent to avoid API blocks
                headers = {'User-Agent': 'HousePredictorProject_v3'}
                geo_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}"
                response = requests.get(geo_url, headers=headers, timeout=5).json()
                
                # Combine everything into one string for keyword matching
                full_address = response.get('display_name', '').lower()
                addr = response.get('address', {})
                
                # Better City Name Logic
                city_display = addr.get('city') or addr.get('town') or addr.get('suburb') or addr.get('district') or "Selected Area"

                # Match Tiers
                prime_list = ['indore', 'ahmedabad', 'mumbai', 'surat', 'pune', 'delhi', 'bangalore']
                standard_list = ['jabalpur', 'bhopal', 'gwalior', 'vadodara', 'rajkot', 'nagpur']

                if any(c in full_address for c in prime_list):
                    tier = 2
                    location_label = "Prime City Area"
                elif any(c in full_address for c in standard_list):
                    tier = 1
                    location_label = "Standard City Area"
                
            except:
                pass

        # 3. Predict (sqft, beds, baths, tier, prop_type, furnish)
        features = np.array([[sqft, beds, baths, tier, prop_type, furnish]])
        val = model.predict(features)[0]
        if request.form.get('amenities'): val *= 1.15

        # 4. Result Formatting
        currency = f"₹{round(val/100, 2)} Crore" if val >= 100 else f"₹{round(val, 2)} Lakh"
        rate = int((val * 100000) / sqft) if sqft > 0 else 0

        return f"""
            <div style='color: #10b981; font-size: 26px; font-weight: 800;'>{currency}</div>
            <div style='color: #94a3b8; font-size: 14px; margin-top: 5px;'>
                <i class='fas fa-map-marker-alt'></i> {city_display.title()} <b>({location_label})</b>
            </div>
            <div style='margin-top: 10px; border-top: 1px solid #334155; padding-top: 10px; font-size: 12px; display: flex; justify-content: space-between;'>
                <span>Rate: ₹{rate}/sqft</span>
                <span>Tier Score: {tier}/2</span>
            </div>
        """
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)