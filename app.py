from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import requests

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Geo-Intelligence Keywords
PREMIUM_KEYS = ['highway', 'nh', 'bypass', 'expressway', 'road', 'marg', 'cloverleaf']
RURAL_KEYS = ['village', 'gram', 'rural', 'kheda', 'basti', 'panchayat']
METROS = ['mumbai', 'delhi', 'bangalore', 'hyderabad', 'pune', 'indore', 'gurgaon', 'noida']
TIER2 = ['jabalpur', 'bhopal', 'gwalior', 'jaipur', 'lucknow', 'nagpur', 'surat']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        req = request.json
        sqft = float(req.get('sqft', 1000))
        beds, baths = float(req.get('beds', 2)), float(req.get('baths', 2))
        tier, mult, loc_label, city = 0, 1.0, "Standard Zone", "Point Selected"

        if req.get('lat') and req.get('lng'):
            headers = {'User-Agent': 'BharatEstate_AI_Pro'}
            res = requests.get(f"https://nominatim.openstreetmap.org/reverse?format=json&lat={req['lat']}&lon={req['lng']}", headers=headers).json()
            addr = res.get('display_name', '').lower()
            city = res.get('address', {}).get('city') or res.get('address', {}).get('town') or "Selected Area"

            if any(k in addr for k in PREMIUM_KEYS):
                mult += 0.22
                loc_label = "High-Growth Connectivity Zone"
            
            if any(c in addr for c in METROS): tier = 2
            elif any(c in addr for c in TIER2): tier = 1
            
            if any(k in addr for k in RURAL_KEYS):
                tier, mult, loc_label = 0, mult - 0.15, "Rural / Village Settlement"

        features = pd.DataFrame([[sqft, beds, baths, tier, float(req.get('prop_type', 0)), float(req.get('furnish', 1))]], 
                                columns=["sqft","beds","baths","tier","prop_type","furnish"])
        
        final_val = max(float(model.predict(features)[0]) * mult, 5.0)

        return jsonify({
            'price': f"₹{final_val/100:.2f} Cr" if final_val >= 100 else f"₹{final_val:.1f} L",
            'rate': f"₹{int((final_val * 100000) / sqft):,}/sqft",
            'label': loc_label, 'city': city.title()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)