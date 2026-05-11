# 🏠 AI House Price Predictor — Bharat Estate AI

A Machine Learning web application that predicts Indian residential property prices using **Linear Regression** (scikit-learn), a Flask backend, and an interactive map-based UI.

---

## 🚀 Features

- **Machine Learning:** Scikit-Learn `LinearRegression` pipeline with `StandardScaler`
- **50-sample dataset** calibrated for 2024–25 Indian real-estate prices
- **6 features:** Area (sqft), Bedrooms, Bathrooms, Location Tier, Property Type, Furnishing
- **Auto Location Detection:** Reverse geocoding via Nominatim API classifies the pinned location into Metro / Tier-2 / Rural tiers
- **Interactive Map:** Leaflet.js map — click to pin, or search any Indian city
- **Confidence Range:** ±12% price band displayed alongside the estimate
- **Backend:** Flask (Python) with JSON API (`/predict` endpoint)
- **Frontend:** Responsive HTML5/CSS3 with DM Sans + Syne typography

---

## 📐 Model Details

| Item | Detail |
|------|--------|
| Algorithm | Linear Regression |
| Library | scikit-learn |
| Training samples | 50 |
| Features | 6 (sqft, beds, baths, tier, prop_type, furnish) |
| R² Score | ~0.91 |
| Output | Price in Lakhs (₹) |

**Equation:**  
`Price = β₀ + β₁·sqft + β₂·beds + β₃·baths + β₄·tier + β₅·prop_type + β₆·furnish`

---

## 🔧 How to Run

```bash
# 1. Clone & navigate
git clone <repo-url>
cd house-price-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (generates model.pkl)
python model.py

# 4. Start the server
python app.py

# 5. Open in browser
http://127.0.0.1:5000
```

---

## 📁 Project Structure

```
house-price-prediction/
├── templates/
│   └── index.html        # Frontend UI
├── app.py                # Flask backend + /predict API
├── model.py              # Linear Regression training script
├── model.pkl             # Saved model (generated)
├── requirements.txt      # Python dependencies
├── Procfile              # Gunicorn config (for deployment)
├── .gitignore
└── README.md
```

---

## 🌐 Deployment (Render / Railway)

The `Procfile` contains:
```
web: gunicorn app:app
```
Set the `PORT` environment variable on your platform. No other changes needed.

---

## 🛠 Tools & Technologies

- Python 3.10+, Flask, scikit-learn, NumPy, pandas
- Leaflet.js, OpenStreetMap Nominatim API
- HTML5, CSS3, Font Awesome 6, Google Fonts
- VS Code, Git