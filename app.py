from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__, template_folder='templates')  # assumes HTML is in ./templates/

# Load trained model
model = joblib.load("results/inlet_xgb_model.joblib")

# Serve the frontend HTML
@app.route("/")
def index():
    return render_template("index.html")  # or the filename of your HTML

# Predict endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    return jsonify({"predicted_wear": float(pred)})

# Documentation endpoint
@app.route("/docs")
def docs():
    with open("docs.md", "r") as f:
        return f.read()

if __name__ == "__main__":
    app.run(debug=True)
