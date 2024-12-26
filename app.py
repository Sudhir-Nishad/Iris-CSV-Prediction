from flask import Flask, request, render_template, send_file
import pandas as pd
import joblib
import os

app = Flask(__name__)
@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico', mimetype='image/vnd.microsoft.icon')

# Load the model and scaler
model = joblib.load('trained_model.pkl')
scaler = joblib.load('scaler.pkl')

UPLOAD_FOLDER = 'uploads'
PREDICTED_FOLDER = 'predicted'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICTED_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded.", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file.", 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Read the uploaded file
    if file.filename.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file.filename.endswith(('.xls', '.xlsx')):
        data = pd.read_excel(file_path)
    else:
        return "Invalid file format. Please upload a CSV or Excel file.", 400

    # Preprocess and predict
    scaled_data = scaler.transform(data)
    predictions = model.predict(scaled_data)
    data['Predictions'] = predictions

    # Save the predicted file
    predicted_file_path = os.path.join(PREDICTED_FOLDER, 'predicted_' + file.filename)
    data.to_csv(predicted_file_path, index=False)

    return send_file(predicted_file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
