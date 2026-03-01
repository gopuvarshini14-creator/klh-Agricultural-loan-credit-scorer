from flask import Flask, render_template
import joblib

app = Flask(__name__)

model = joblib.load("models/credit_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run()
