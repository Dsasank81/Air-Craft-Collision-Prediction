✈️ Aircraft Collision Prediction System
Machine Learning Model + Flask Web Application

This project is a Machine Learning–powered Aircraft Collision Risk Prediction System that analyzes flight sensor data such as speed, altitude, pressure, and distance to the nearest aircraft to determine whether the aircraft is at High Collision Risk or Low Collision Risk.

The system includes:

A model training pipeline (using multiple ML algorithms and an ensemble classifier)

A Flask web application for real-time prediction

Saved model files for deployment

📂 Project Structure
├── train_model.py               # Machine learning model training script
├── app.py                       # Flask application for prediction
├── aircraft_collision_model.pkl # Trained ensemble model
├── scaler.pkl                   # StandardScaler for feature normalization
├── templates/
│   └── index.html               # Web UI page
└── README.md                    # Project documentation

🚀 Features
✔ Machine Learning

Ensemble model using VotingClassifier with:
GaussianNB, Logistic Regression, KNN, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, and MLP.

Feature scaling using StandardScaler

Accuracy, confusion matrix, classification report, cross-validation

Saves trained model as .pkl files

✔ Web Application (Flask)

Simple interface to input flight parameters

Real-time collision risk prediction

Clean UI with error handling for invalid inputs

🧠 How the Model Works
1. Inputs Used

Speed

Altitude

Pressure

Distance to nearest aircraft

2. Output

High Collision Risk (1)

Low Collision Risk (0)

3. Algorithms Used

An ensemble model with soft voting improves accuracy and prediction stability.

4. Model Saving

aircraft_collision_model.pkl and scaler.pkl are saved for deployment.

🛠️ Installation & Setup
1. Clone the Repository
git clone https://github.com/<your-username>/aircraft-collision-prediction.git
cd aircraft-collision-prediction

2. Install Dependencies
pip install -r requirements.txt

3. Train the Model (optional)
python train_model.py

4. Run the Flask App
python app.py


Open browser and visit:
👉 http://127.0.0.1:5000/

🌐 Web Application Overview

The Flask app:

Takes user input

Scales values using scaler.pkl

Predicts collision risk using the machine learning model

Displays either:

🔴 High Collision Risk

🟢 Low Collision Risk

📈 Results & Evaluation

The system prints:

Model accuracy

Classification report

Confusion matrix

Cross-validation performance

(Values depend on the dataset used.)

🚀 Future Improvements

Use real-world ADS-B flight data

Implement deep learning (LSTMs) for time-series prediction

Add alerting system

Host the app on cloud (AWS / Azure / GCP)

Explainable AI using SHAP

🤝 Contributing

Pull requests are welcome!
Feel free to open an issue for suggestions or bugs.

📜 License

This project is licensed under the MIT License.
