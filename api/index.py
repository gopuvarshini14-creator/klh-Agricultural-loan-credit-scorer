
from flask import Flask, render_template
import joblib

app = Flask(__name__,
            template_folder="../templates",
            static_folder="../static")

# Load model
model = joblib.load("../models/credit_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

handler = app
