from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        type_val = int(request.form["type"])
        air = float(request.form["air_temp"])
        proc = float(request.form["proc_temp"])
        rpm = float(request.form["rpm"])
        torque = float(request.form["torque"])
        wear = float(request.form["tool_wear"])

        data = np.array([[type_val, air, proc, rpm, torque, wear]])
        data = scaler.transform(data)

        pred = model.predict_proba(data)[0][1]
        prediction = pred

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)