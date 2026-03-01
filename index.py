from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("models/loan_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # frontend must send JSON
        
        # Convert input values to float
        features = [float(value) for value in data.values()]
        
        prediction = model.predict([features])
        
        return jsonify({
            "prediction": int(prediction[0])
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run()
