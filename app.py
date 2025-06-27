from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Global variable to store the trained model
model = None

def load_or_train_model():
    """Load existing model or train a new one"""
    global model
    model_path = 'heart_disease_model.pkl'
    
    if os.path.exists(model_path):
        # Load existing model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded from file")
    else:
        # Train new model
        print("Training new model...")
        heart_data = pd.read_csv('heart_disease_data.csv')
        
        # Prepare features and target
        X = heart_data.drop(columns='target', axis=1)
        Y = heart_data['target']
        
        # Split data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
        
        # Train model
        model = LogisticRegression()
        model.fit(X_train, Y_train)
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print("Model trained and saved")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = request.get_json()
        
        # Extract features in the correct order
        features = [
            float(data['age']),
            int(data['sex']),
            int(data['cp']),
            float(data['trestbps']),
            float(data['chol']),
            int(data['fbs']),
            int(data['restecg']),
            float(data['thalach']),
            int(data['exang']),
            float(data['oldpeak']),
            int(data['slope']),
            int(data['ca']),
            int(data['thal'])
        ]
        
        # Convert to numpy array and reshape
        input_data = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        # Prepare response
        result = {
            'prediction': int(prediction),
            'probability_healthy': round(probability[0] * 100, 2),
            'probability_disease': round(probability[1] * 100, 2),
            'message': 'The person has heart disease' if prediction == 1 else 'The person does not have heart disease'
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/info')
def get_info():
    """Get information about the model and features"""
    feature_info = {
        'age': 'Age in years',
        'sex': 'Sex (1 = male; 0 = female)',
        'cp': 'Chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic)',
        'trestbps': 'Resting blood pressure (mm Hg)',
        'chol': 'Serum cholesterol (mg/dl)',
        'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)',
        'restecg': 'Resting electrocardiographic results (0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy)',
        'thalach': 'Maximum heart rate achieved',
        'exang': 'Exercise induced angina (1 = yes; 0 = no)',
        'oldpeak': 'ST depression induced by exercise relative to rest',
        'slope': 'Slope of the peak exercise ST segment (0: upsloping, 1: flat, 2: downsloping)', 
        'ca': 'Number of major vessels colored by fluoroscopy (0-3)',
        'thal': 'Thalassemia (1: normal, 2: fixed defect, 3: reversible defect)'
    }
    
    return jsonify({
        'features': feature_info,
        'target': 'Heart disease (1 = disease, 0 = no disease)'
    })

if __name__ == '__main__':
    load_or_train_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
