# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the logistic regression model
with open('finalized_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define routes
@app.route('/')
def index():
    # Render the HTML template with buttons for each output
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the inputs from the form
    inputs = [float(request.form[f'input_{i}']) for i in range(1, 21)]
    
    # Convert inputs to numpy array
    inputs_np = np.array([inputs])

    # Use the model to predict
    prediction = model.predict(inputs_np)
    
    # Return the result
    return f'Prediction: {prediction[0]}'

if __name__ == '__main__':
    app.run(debug=True)
