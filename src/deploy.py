# deploy.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load("src/model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    """Predict readmission risk for a patient."""
    data = request.get_json()
    df = pd.DataFrame([data])
    
    # Ensure required features are present
    required_features = model.feature_names_in_
    df = df[required_features]
    
    # Make prediction
    prediction = model.predict_proba(df)[:, 1][0]
    
    return jsonify({"readmission_risk": float(prediction)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
