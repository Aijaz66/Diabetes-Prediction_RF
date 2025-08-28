from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load saved RF model
model = joblib.load("rf_rus_model.pkl")

# Smoking map
smoking_map = {
    "never": 0,
    "former": 1,
    "not current": 2,
    "ever": 3,
    "current": 4
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Get form inputs
            age = float(request.form["age"])
            bmi = float(request.form["bmi"])
            hba1c = float(request.form["hba1c"])
            glucose = float(request.form["glucose"])
            gender = request.form["gender"]
            hypertension = int(request.form["hypertension"])
            heart_disease = int(request.form["heart_disease"])
            smoking_history = request.form["smoking"]

            # Map smoking string to number
            smoking_risk_level = smoking_map[smoking_history]

            # Create DataFrame
            input_data = pd.DataFrame([{
                "age": age,
                "bmi": bmi,
                "HbA1c_level": hba1c,
                "blood_glucose_level": glucose,
                "gender": gender,
                "hypertension": hypertension,
                "heart_disease": heart_disease,
                "smoking_risk_level": smoking_risk_level
            }])

            # Predict
            proba = model.predict_proba(input_data)[0][1]
            prediction = 1 if proba >= 0.65 else 0   # ✅ threshold 65%

            result = "✅ High Risk of Diabetes" if prediction == 1 else "❌ Low Risk of Diabetes"
            return render_template("index.html", result=result, probability=f"{proba:.2%}")

        except Exception as e:
            return render_template("index.html", result=f"⚠️ Error: {str(e)}", probability=None)

    return render_template("index.html", result=None, probability=None)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
