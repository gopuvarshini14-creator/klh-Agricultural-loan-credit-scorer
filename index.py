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
        data = request.json

        # Use training feature order automatically
        features = [float(data[col]) for col in credit_model.feature_names_in_]
        input_data = np.array([features])

        if hasattr(credit_model, "predict_proba"):
            probability = credit_model.predict_proba(input_data)[0][1]
            credit_score = int(probability * 100)
        else:
            credit_score = int(credit_model.predict(input_data)[0])

        eligible_loan = int(loan_model.predict(input_data)[0])

        return jsonify({
            "credit_score": credit_score,
            "eligible_loan": eligible_loan
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run()
