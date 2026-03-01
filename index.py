from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the correct model
model = joblib.load("models/credit_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    
    features = [float(x) for x in data.values()]
    
    prediction = model.predict([features])
    
    return jsonify({
        "prediction": int(prediction[0])
    })

if __name__ == "__main__":
    app.run()
