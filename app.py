from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the models
diabetes_model = pickle.load(open('E:\datasets\diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('E:\datasets\heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('E:\datasets\parkinsons_model.sav', 'rb'))

# Homepage route
@app.route("/")
def home():
    return render_template("home.html")

# Diabetes prediction page
@app.route("/diabetesprediction", methods=["POST", "GET"])
def diabetesprediction():
    if request.method == "POST":
        input_data = (
            int(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['bloodPressure']),
            float(request.form['skinThickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['pedigreeFunction']),
            int(request.form['age'])
        )

        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        prediction = diabetes_model.predict(input_data_reshaped)

        result = "The person is diabetic" if prediction[0] == 1 else "The person is not diabetic"
        return render_template("dia_result.html", result=result)

    return render_template("diabetes.html")

# Heart Disease prediction page
@app.route("/heartdiseaseprediction", methods=["POST", "GET"])
def heartdiseaseprediction():
    if request.method == "POST":
        input_data = (
            int(request.form['age']),
            int(request.form['sex']),
            float(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            float(request.form['fbs']),
            float(request.form['restecg']),
            float(request.form['thalach']),
            int(request.form['exang']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            int(request.form['ca']),
            int(request.form['thal'])
        )

        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        prediction = heart_disease_model.predict(input_data_reshaped)

        result = "The Person has Heart Disease" if prediction[0] == 1 else "The Person does not have a Heart Disease"
        return render_template("heart_result.html", result=result)

    return render_template("heart.html")

# Parkinson's Disease prediction page
@app.route("/prankinsonsdiseaseprediction", methods=["POST", "GET"])
def prankinsonsdiseaseprediction():
    if request.method == "POST":
        input_data = (
            float(request.form['mdvp_fo']),
            float(request.form['mdvp_fhi']),
            float(request.form['mdvp_flo']),
            float(request.form['mdvp_jitter']),
            float(request.form['mdvp_jitter_abs']),
            float(request.form['mdvp_rap']),
            float(request.form['mdvp_ppq']),
            float(request.form['jitter_ddp']),
            float(request.form['mdvp_shimmer']),
            float(request.form['mdvp_shimmer_db']),
            float(request.form['shimmer_apq3']),
            float(request.form['shimmer_apq5']),
            float(request.form['mdvp_apq']),
            float(request.form['shimmer_dda']),
            float(request.form['nhr']),
            float(request.form['hnr']),
            float(request.form['rpde']),
            float(request.form['dfa']),
            float(request.form['spread1']),
            float(request.form['spread2']),
            float(request.form['d2']),
            float(request.form['ppe'])
        )

        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        prediction = parkinsons_model.predict(input_data_reshaped)

        result = "The Person has Parkinsons" if prediction[0] == 1 else "The Person does not have Parkinsons Disease"
        return render_template("prankinsons_result.html", result=result)

    return render_template("prankinsons.html")

# Health Guard page
@app.route("/healthguard")
def healthguard():
    return render_template("healthguard.html")

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)