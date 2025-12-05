import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Load the trained model and scaler
try:
    model = joblib.load('aircraft_collision_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    print("Error: Model files not found. Make sure 'aircraft_collision_model.pkl' and 'scaler.pkl' are in the correct directory.")
    exit()


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get input data from the form
            speed = float(request.form['speed'])
            altitude = float(request.form['altitude'])
            pressure = float(request.form['pressure'])
            distance = float(request.form['distance'])

            # Prepare the input data
            input_data = pd.DataFrame({
                'speed': [speed],
                'altitude': [altitude],
                'pressure': [pressure],
                'distance_to_nearest_aircraft': [distance]
            })

            # Scale the input data using the same scaler used during training
            numerical_features = input_data.select_dtypes(include=np.number).columns
            input_data[numerical_features] = scaler.transform(input_data[numerical_features])

            # Make prediction
            prediction = model.predict(input_data)[0]

            # Determine the result message
            if prediction == 1:
                result_message = "High Collision Risk"
            else:
                result_message = "Low Collision Risk"

            return render_template('index.html', result=result_message)

        except ValueError:
            return render_template('index.html', result="Invalid input. Please enter numerical values.")

        except Exception as e:
            print(f"An error occurred: {e}")  # Log the exception for debugging
            return render_template('index.html', result="An error occurred. Please try again.")

    # Render the initial form on GET request
    return render_template('index.html', result=None)


if __name__ == '__main__':
    app.run(debug=True)
