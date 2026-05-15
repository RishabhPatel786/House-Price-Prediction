from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import requests
import os

app = Flask(__name__)

# Load the Linear Regression Pipeline
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Capture Inputs
        sqft = float(request.form.get('sqft', 1000))
        bhk = int(request.form.get('bhk', 2))
        age = int(request.form.get('age', 5))
        floor = int(request.form.get('floor', 1))
        road_width = int(request.form.get('road_width', 30))
        d_road = float(request.form.get('dist_road', 0.5))
        d_metro = float(request.form.get('dist_metro', 5.0))
        
        # 2. Tier Detection via Map
        tier = 0
        lat, lng = request.form.get('lat'), request.form.get('lng')
        if lat and lng:
            res = requests.get(f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}", headers={'User-Agent':'BE_AI'}).json()
            addr = res.get('display_name', '').lower()
            if any(c in addr for c in ['mumbai', 'delhi', 'bangalore', 'pune', 'indore', 'hyderabad']): tier = 2
            elif any(c in addr for c in ['jabalpur', 'bhopal', 'gwalior', 'lucknow', 'nagpur']): tier = 1

        # 3. Features for Linear Model
        features = pd.DataFrame([[sqft, bhk, age, tier, d_road, d_metro, 
                                 int(request.form.get('furnish', 1)), 
                                 floor, int(request.form.get('type', 0)), road_width]], 
                                columns=['sqft','bhk','age','tier','dist_road','dist_metro','furnish','floor','type','road_width'])
        
        # Base Linear Prediction
        prediction = model.predict(features)[0]

        # 4. Expert Adjustments (Manual Multipliers for accuracy)
        multiplier = 1.0
        if request.form.get('amenities') == 'on': multiplier += 0.15
        if request.form.get('gated') == 'on': multiplier += 0.12
        if request.form.get('main_road_touch') == 'on': multiplier += 0.20
        
        # Apply Distance-based impact from your table
        if d_road > 2.0: multiplier *= 0.60  # 40% discount for interior
        elif d_road < 0.1: multiplier *= 1.15 # 15% premium for road touch
        
        final_val = max(prediction * multiplier, 8.5) # Inflation floor

        return jsonify({
            'price': f"₹{final_val/100:.2f} Cr" if final_val >= 100 else f"₹{final_val:.1f} L",
            'rate': f"₹{int((final_val * 100000) / sqft):,}/sqft",
            'tier': tier
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)