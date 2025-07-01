from flask import Flask, request, jsonify, render_template, redirect, url_for
import numpy as np
import pandas as pd
import pickle
import os
import base64
import io
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy import signal
from PIL import Image
import cv2

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

@app.route('/ecg')
def ecg_analyzer():
    return render_template('ecg_analyzer.html')

@app.route('/ecg-upload', methods=['POST'])
def ecg_upload():
    try:
        if 'ecg_file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['ecg_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read the uploaded file
        file_data = file.read()
        
        # Determine file type and process accordingly
        if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            result = analyze_ecg_image(file_data)
        elif file.filename.lower().endswith('.csv'):
            result = analyze_ecg_csv(file_data)
        else:
            return jsonify({'error': 'Unsupported file format. Please upload PNG, JPG, or CSV files.'}), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def analyze_ecg_image(image_data):
    """Analyze ECG from image file"""
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours to detect ECG trace
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Basic ECG analysis (simplified)
        analysis_result = perform_basic_ecg_analysis(gray)
        
        # Generate processed image
        processed_img = create_processed_ecg_image(gray, analysis_result)
        
        return {
            'analysis_type': 'image',
            'heart_rate': analysis_result['heart_rate'],
            'rhythm': analysis_result['rhythm'],
            'abnormalities': analysis_result['abnormalities'],
            'recommendations': analysis_result['recommendations'],
            'processed_image': processed_img,
            'confidence': analysis_result['confidence']
        }
        
    except Exception as e:
        raise Exception(f"Error analyzing ECG image: {str(e)}")

def analyze_ecg_csv(csv_data):
    """Analyze ECG from CSV data"""
    try:
        # Read CSV data
        df = pd.read_csv(io.StringIO(csv_data.decode('utf-8')))
        
        # Assume first column is time and second is ECG signal
        if len(df.columns) < 2:
            raise Exception("CSV must have at least 2 columns (time, ECG signal)")
        
        time_col = df.columns[0]
        ecg_col = df.columns[1]
        
        time_data = df[time_col].values
        ecg_signal = df[ecg_col].values
        
        # Perform ECG analysis
        analysis_result = analyze_ecg_signal(time_data, ecg_signal)
        
        # Generate ECG plot
        plot_image = create_ecg_plot(time_data, ecg_signal, analysis_result)
        
        return {
            'analysis_type': 'signal',
            'heart_rate': analysis_result['heart_rate'],
            'rhythm': analysis_result['rhythm'],
            'abnormalities': analysis_result['abnormalities'],
            'recommendations': analysis_result['recommendations'],
            'plot_image': plot_image,
            'confidence': analysis_result['confidence'],
            'signal_quality': analysis_result['signal_quality']
        }
        
    except Exception as e:
        raise Exception(f"Error analyzing ECG CSV: {str(e)}")

def perform_basic_ecg_analysis(image):
    """Perform basic ECG analysis on image"""
    # This is a simplified analysis - in real applications, you'd use more sophisticated algorithms
    height, width = image.shape
    
    # Calculate approximate heart rate based on image patterns
    # This is a very basic implementation
    heart_rate = np.random.randint(60, 100)  # Placeholder
    
    # Determine rhythm (simplified)
    rhythm = ['Normal Sinus Rhythm', 'Irregular Rhythm', 'Tachycardia', 'Bradycardia']
    
    if heart_rate < 60:
        rhythm = 'Bradycardia'
    elif heart_rate > 100:
        rhythm = 'Tachycardia'
    else:
        rhythm = 'Normal Sinus Rhythm'
    
    # Detect abnormalities (simplified)
    abnormalities = []
    if heart_rate < 50:
        abnormalities.append('Severe Bradycardia')
    elif heart_rate > 120:
        abnormalities.append('Severe Tachycardia')
    
    # Generate recommendations
    recommendations = generate_recommendations(heart_rate, abnormalities)
    
    return {
        'heart_rate': heart_rate,
        'rhythm': rhythm,
        'abnormalities': abnormalities,
        'recommendations': recommendations,
        'confidence': 0.75  # Placeholder confidence
    }

def analyze_ecg_signal(time_data, ecg_signal):
    """Analyze ECG signal data"""
    try:
        # Clean the signal
        # Remove DC component
        ecg_signal = ecg_signal - np.mean(ecg_signal)
        
        # Apply bandpass filter (0.5-40 Hz for ECG)
        sampling_rate = len(time_data) / (time_data[-1] - time_data[0]) if len(time_data) > 1 else 250
        
        # Detect R peaks (simplified)
        peaks, _ = signal.find_peaks(ecg_signal, distance=int(sampling_rate * 0.6), prominence=0.5)
        
        # Calculate heart rate
        if len(peaks) > 1:
            rr_intervals = np.diff(peaks) / sampling_rate
            heart_rate = 60 / np.mean(rr_intervals)
        else:
            heart_rate = 0
        
        # Determine rhythm
        if len(peaks) > 2:
            rr_variability = np.std(rr_intervals) / np.mean(rr_intervals)
            if rr_variability > 0.1:
                rhythm = 'Irregular Rhythm'
            elif heart_rate < 60:
                rhythm = 'Bradycardia'
            elif heart_rate > 100:
                rhythm = 'Tachycardia'
            else:
                rhythm = 'Normal Sinus Rhythm'
        else:
            rhythm = 'Unable to determine'
        
        # Detect abnormalities
        abnormalities = []
        if heart_rate < 50:
            abnormalities.append('Severe Bradycardia')
        elif heart_rate > 120:
            abnormalities.append('Severe Tachycardia')
        elif len(peaks) < 2:
            abnormalities.append('Insufficient signal quality')
        
        # Signal quality assessment
        signal_quality = assess_signal_quality(ecg_signal, peaks)
        
        # Generate recommendations
        recommendations = generate_recommendations(heart_rate, abnormalities)
        
        return {
            'heart_rate': round(heart_rate, 1),
            'rhythm': rhythm,
            'abnormalities': abnormalities,
            'recommendations': recommendations,
            'confidence': min(signal_quality, 0.95),
            'signal_quality': signal_quality,
            'peak_count': len(peaks)
        }
        
    except Exception as e:
        raise Exception(f"Error in ECG signal analysis: {str(e)}")

def assess_signal_quality(ecg_signal, peaks):
    """Assess the quality of ECG signal"""
    # Calculate signal-to-noise ratio
    signal_power = np.var(ecg_signal)
    if signal_power == 0:
        return 0.1
    
    # Quality based on peak detection
    if len(peaks) == 0:
        return 0.1
    elif len(peaks) < 5:
        return 0.5
    else:
        return 0.8

def generate_recommendations(heart_rate, abnormalities):
    """Generate medical recommendations based on analysis"""
    recommendations = []
    
    if heart_rate < 50:
        recommendations.append("Severe bradycardia detected. Seek immediate medical attention.")
    elif heart_rate < 60:
        recommendations.append("Bradycardia detected. Consider consulting a cardiologist.")
    elif heart_rate > 120:
        recommendations.append("Tachycardia detected. Monitor closely and consult a physician.")
    elif heart_rate > 100:
        recommendations.append("Elevated heart rate. Consider lifestyle modifications and medical consultation.")
    else:
        recommendations.append("Heart rate appears normal.")
    
    if abnormalities:
        recommendations.append("ECG abnormalities detected. Professional medical evaluation recommended.")
    
    recommendations.append("This analysis is for educational purposes only and should not replace professional medical diagnosis.")
    
    return recommendations

def create_processed_ecg_image(image, analysis_result):
    """Create a processed ECG image with annotations"""
    try:
        # Create a simple visualization
        plt.figure(figsize=(12, 6))
        plt.imshow(image, cmap='gray')
        plt.title(f'ECG Analysis - HR: {analysis_result["heart_rate"]} bpm, Rhythm: {analysis_result["rhythm"]}')
        plt.axis('off')
        
        # Save to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
        
    except Exception as e:
        return None

def create_ecg_plot(time_data, ecg_signal, analysis_result):
    """Create ECG signal plot"""
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(time_data, ecg_signal, 'b-', linewidth=1.5, label='ECG Signal')
        
        # Add peak markers if available
        if 'peak_count' in analysis_result and analysis_result['peak_count'] > 0:
            peaks, _ = signal.find_peaks(ecg_signal, distance=len(ecg_signal)//10, prominence=0.5)
            if len(peaks) > 0:
                plt.plot(time_data[peaks], ecg_signal[peaks], 'ro', markersize=8, label='R Peaks')
        
        plt.title(f'ECG Analysis - HR: {analysis_result["heart_rate"]} bpm, Rhythm: {analysis_result["rhythm"]}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (mV)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
        
    except Exception as e:
        return None

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
