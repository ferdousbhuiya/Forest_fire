from flask import Flask, render_template, jsonify, request, flash
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Import ridge regression and StandardScaler pickle models
try:
    ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
    scaler_model = pickle.load(open('models/scaler.pkl', 'rb'))
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    ridge_model = None
    scaler_model = None

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))

            # Print to check input values
            print(f"Received: Temperature={Temperature}, RH={RH}, Ws={Ws}, Rain={Rain}, FFMC={FFMC}, DMC={DMC}, ISI={ISI}, Classes={Classes}, Region={Region}")
            
            # Scale the new data
            new_data_scaled = scaler_model.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])

            # Print to check scaled data
            print(f"Scaled data: {new_data_scaled}")

            # Predict using the ridge model
            result = ridge_model.predict(new_data_scaled)

            # Print to check result
            print(f"Prediction result: {result}")
            
            return render_template('home.html', results=result[0])
        except Exception as e:
            print(f"Error during prediction: {e}")
            flash('Error during prediction. Please check your input values.')
            return render_template('home.html')
    else:
        return render_template('home.html')
    
if __name__ == "__main__":
    app.run(port=5001, debug=True)
