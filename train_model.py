import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 1. Data Preparation
# Sample data creation (replace this with your actual DataFrame)
data = {
    'speed': [70, 85, 60, 90, 75, 80, 95, 100],
    'altitude': [10000, 12000, 9000, 11000, 9500, 11500, 10500, 13000],
    'pressure': [1013, 1010, 1015, 1005, 1012, 1008, 1007, 1006],
    'distance_to_nearest_aircraft': [15, 10, 20, 5, 12, 8, 3, 2],
    'collision': [0, 1, 0, 1, 0, 1, 1, 1]
}

# Create DataFrame
df = pd.DataFrame(data)

# Handle missing values (if any)
df = df.dropna()

# Separate features and target variable
X = df.drop('collision', axis=1)
y = df['collision']

# 2. Data Preprocessing
# Scale the numerical features
numerical_features = X.select_dtypes(include=np.number).columns
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Model Training and Ensemble
# Individual Models

model1 = GaussianNB()
model2 = LogisticRegression(random_state=42)
model3 = KNeighborsClassifier(n_neighbors=4)
model4 = DecisionTreeClassifier(random_state=42)
model5 = RandomForestClassifier(n_estimators=100, random_state=42)
model6 = GradientBoostingClassifier(n_estimators=100, random_state=42)
model7 = AdaBoostClassifier(n_estimators=100)
model8 = MLPClassifier(random_state=42)

# Ensemble with VotingClassifier
voting_clf = VotingClassifier(estimators=[
    ('gnb', model1), ('lr', model2), ('knn', model3), ('dt', model4),
    ('rf', model5), ('gb', model6), ('ada', model7), ('mlp', model8)
], voting='soft') 
voting_clf.fit(X_train, y_train)

# 5. Evaluation
# Predictions
y_pred = voting_clf.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100 }")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Cross-Validation
kfold = StratifiedKFold(n_splits=4)
cv_results = cross_val_score(voting_clf, X_train, y_train, cv=kfold)
print(f"Cross-Validation Mean Accuracy: {cv_results.mean()}")

# Save the Model and Scaler
joblib.dump(voting_clf, 'aircraft_collision_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully.")
