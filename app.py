from flask import Flask, render_template, request
import pickle
import numpy as np
import requests
import os

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Inputs
        sqft = float(request.form.get('sqft', 1000))
        beds = float(request.form.get('beds', 2))
        baths = float(request.form.get('baths', 2))
        prop_type = float(request.form.get('prop_type', 0))
        furnish = float(request.form.get('furnish', 0))
        lat, lng = request.form.get('lat'), request.form.get('lng')

        tier, multiplier, loc_label, city_name = 0, 1.0, "Rural Area", "Selected Location"

        if lat and lng:
            headers = {'User-Agent': 'BharatEstate_v8'}
            res = requests.get(f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}", headers=headers).json()
            full_addr = res.get('display_name', '').lower()
            city_name = res.get('address', {}).get('city') or res.get('address', {}).get('town') or "Selected Area"

            # 2026 Market Multipliers
            if any(c in full_addr for c in ['mumbai', 'delhi', 'bangalore', 'pune', 'indore', 'ahmedabad']):
                tier, multiplier, loc_label = 2, 1.8, "Prime Metro Zone"
            elif any(c in full_addr for c in ['jabalpur', 'bhopal', 'gwalior', 'jaipur', 'lucknow']):
                tier, multiplier, loc_label = 1, 1.25, "Standard City Area"

        # AI Prediction
        features = np.array([[sqft, beds, baths, tier, prop_type, furnish]])
        pred = model.predict(features)[0]
        
        # Apply Multiplier & Floor Price Guardrails
        final_val = pred * multiplier
        if tier == 2: final_val = max(final_val, 55.0) # Metro Floor
        elif tier == 1: final_val = max(final_val, 28.0) # City Floor
        else: final_val = max(final_val, 8.5) # Rural Floor

        currency = f"₹{final_val/100:.2f} Crore" if final_val >= 100 else f"₹{final_val:.1f} Lakh"
        rate = int((final_val * 100000) / sqft)

        return f"""
            <div style='background:rgba(99,102,241,0.1); padding:20px; border-radius:15px; border:1px solid #6366f1;'>
                <div style='color:#6366f1; font-size:12px; font-weight:800; text-transform:uppercase;'>Market Valuation</div>
                <div style='color:#10b981; font-size:32px; font-weight:900;'>{currency}</div>
                <div style='color:#94a3b8; font-size:14px; margin-top:5px;'>{city_name.title()} | <b>{loc_label}</b></div>
                <div style='margin-top:15px; font-size:12px; color:#64748b; border-top:1px solid rgba(255,255,255,0.05); padding-top:10px;'>
                    Avg Rate: ₹{rate:,}/sqft
                </div>
            </div>
        """
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)