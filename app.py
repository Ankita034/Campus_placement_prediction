from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import joblib

app = Flask(__name__)

# Load the pre-trained neural network model
model = keras.models.load_model("placement_model.keras")

# Load the StandardScaler
scaler = joblib.load("scaler.pkl")

# Define a route for the home page
@app.route('/')
def home():
    return render_template('front.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data and convert to DataFrame
    form_data = {
        "CGPA": float(request.form['CGPA']),
        "Internships": int(request.form['Internships']),
        "Projects": int(request.form['Projects']),
        "Workshops/Certifications": int(request.form['Workshops/Certifications']),
        "AptitudeTestScore": float(request.form['AptitudeTestScore']),
        "SoftSkillsRating": int(request.form['SoftSkillsRating']),
        "ExtracurricularActivities": int(request.form['ExtracurricularActivities']),
        "PlacementTraining": int(request.form['PlacementTraining']),
        "SSC_Marks": float(request.form['SSC_Marks']),
        "HSC_Marks": float(request.form['HSC_Marks'])
    }
    features = pd.DataFrame([form_data])

    # Scale the features
    scaled_features = scaler.transform(features)

    # Make prediction
    prediction = model.predict(scaled_features)

    # Map prediction to meaningful result
    result = "Placed" if prediction > 0.5 else "Not Placed"
    pred = float(prediction[0])

    print("Pred:", pred)

    # Return prediction result
    return render_template('home.html', pred=pred)

if __name__ == '__main__':
    app.run(debug=True)
