# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 08:10:36 2025

@author: arjun
"""

# app.py
from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Load the trained logistic regression model
with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "🚦 Accident Severity Classifier API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input JSON data
        input_data = request.get_json()

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0].tolist()

        return jsonify({
            'prediction': prediction,
            'prediction_proba': prediction_proba
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("📢 Flask is starting...")
    app.run(debug=True)
