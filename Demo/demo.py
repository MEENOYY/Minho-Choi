from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
import webbrowser
from threading import Timer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the saved model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to open browser automatically
def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try: 
        file = request.files['file']
        data = pd.read_csv(file)
        selected_features = ['thalach', 'oldpeak', 'ca', 'chol', 'thal']
        data.dropna(subset=selected_features, inplace=True)
        X = data[selected_features]
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = pd.DataFrame(X)
        
        # Make predictions
        predictions = model.predict(X)
        predictions = [int(i) for i in predictions]
        prediction_dict = {counter: prediction for counter, prediction in enumerate(predictions, start=1)}
        # Create response
        return jsonify(prediction_dict)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Open browser after a short delay
    Timer(1.5, open_browser).start()
    # Run the Flask app
    app.run(port=5000)
        
