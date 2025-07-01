# Heart Disease Prediction & ECG Analyzer

A comprehensive web application for cardiovascular health assessment using machine learning and electrocardiogram (ECG) analysis.

## Features

### 1. Heart Disease Prediction
- **Machine Learning Model**: Logistic Regression trained on cardiovascular risk factors
- **13 Clinical Parameters**: Age, sex, chest pain type, blood pressure, cholesterol, etc.
- **Real-time Prediction**: Instant risk assessment with probability scores
- **Interactive Web Interface**: User-friendly form with validation

### 2. ECG Analyzer
- **Multi-format Support**: Accepts PNG, JPG, and CSV files
- **Automated Analysis**: AI-powered ECG interpretation
- **Heart Rate Detection**: Automatic BPM calculation
- **Rhythm Classification**: Normal sinus rhythm, tachycardia, bradycardia detection
- **Abnormality Detection**: Identifies potential cardiac irregularities
- **Signal Processing**: Advanced filtering and peak detection algorithms
- **Visual Results**: Processed ECG images with annotations

## Technologies Used

- **Backend**: Flask (Python)
- **Machine Learning**: Scikit-learn, NumPy, Pandas
- **Signal Processing**: SciPy, OpenCV
- **Visualization**: Matplotlib
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **File Processing**: PIL (Pillow)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/heart_disease_prediction_model.git
   cd heart_disease_prediction_model
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Access the application**
   - Open your browser and go to `http://localhost:5000`
   - Heart Disease Prediction: `http://localhost:5000/`
   - ECG Analyzer: `http://localhost:5000/ecg`

## Usage

### Heart Disease Prediction
1. Fill in the medical parameters in the form
2. Click "Predict Heart Disease Risk"
3. View the prediction results with probability scores

### ECG Analyzer
1. Navigate to the ECG Analyzer page
2. Upload an ECG file (PNG, JPG, or CSV format)
3. Click "Analyze ECG" 
4. Review the automated analysis results including:
   - Heart rate (BPM)
   - Rhythm classification
   - Detected abnormalities
   - Medical recommendations
   - Confidence scores

## File Formats

### ECG Image Files (PNG, JPG)
- Clear ECG traces with good contrast
- Higher resolution images provide better analysis
- Multiple lead recordings supported

### ECG CSV Files
- **Column 1**: Time values (in seconds)
- **Column 2**: ECG signal amplitude (in mV)
- **Example**: See `sample_ecg_data.csv`

## Model Information

### Heart Disease Prediction Model
- **Algorithm**: Logistic Regression
- **Training Data**: Heart disease dataset with 303 samples
- **Features**: 13 clinical parameters
- **Accuracy**: ~85% on test data

### ECG Analysis Features
- **Peak Detection**: R-wave identification using SciPy
- **Heart Rate Calculation**: Based on R-R intervals
- **Signal Quality Assessment**: Automated quality scoring
- **Rhythm Analysis**: Classification of cardiac rhythms
- **Filtering**: Bandpass filtering (0.5-40 Hz)

## API Endpoints

- `GET /` - Heart disease prediction interface
- `GET /ecg` - ECG analyzer interface
- `POST /predict` - Heart disease prediction API
- `POST /ecg-upload` - ECG analysis API
- `GET /api/info` - Model feature information

## Medical Disclaimer

⚠️ **Important**: This application is for educational and research purposes only. The results should not be used for medical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Future Enhancements

- [ ] Support for more ECG file formats (EDF, MIT-BIH)
- [ ] Advanced arrhythmia detection
- [ ] Multi-lead ECG analysis
- [ ] Integration with wearable devices
- [ ] Real-time ECG monitoring
- [ ] Export analysis reports
- [ ] Database integration for patient records