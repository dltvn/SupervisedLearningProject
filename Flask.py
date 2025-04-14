from groupProject import FeatureSubsetSelector
from flask_cors import CORS
# app.py
from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Load the trained logistic regression model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)
CORS(app)
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
