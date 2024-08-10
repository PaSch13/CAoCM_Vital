import joblib
import numpy as np

# Load the model and scaler
gb_clf_loaded = joblib.load('CaoCom_Model_StaticData/model_classifier/gradient_boosting_model.pkl')
scaler_loaded = joblib.load('CaoCom_Model_StaticData/model_classifier/scaler.pkl')

# Function to preprocess new data
def preprocess_new_data(data):
    # Convert 'm' to 1 and 'f' to 0
    data = [1 if x == 'm' else 0 if x == 'f' else x for x in data]
    X_new_scaled = scaler_loaded.transform([data])  # Ensure data is in a 2D array
    return X_new_scaled

# 'age', 'sex', 'height', 'weight', 'bmi', 'asa', 'emop'
data = [
    [72, 'm', 160, 60, 22.7, 2, 1],
    [17, 'f', 170, 56.8, 22.7, 2, 1],
    [78, 'f', 154, 56.8, 22.7, 2, 1],
    [39, 'f', 190, 56.8, 22.7, 2, 1],
    [20, 'm', 186, 56.8, 22.7, 2, 1],
    [64, 'm', 180, 56.8, 22.7, 2, 1],
    [53, 'm', 172, 56.8, 22.7, 2, 1]
]
age = 23
height = 190
weight = 70.4
gender = 'f'
asa = 1
emop = 0
bmi = weight/((height/100)**2)
print(bmi)

print(age, height, weight, gender, asa, emop, bmi)

new_data = [age, gender, height, weight, bmi, asa, emop]


# Preprocess and predict
X_new_scaled = preprocess_new_data(new_data)
predictions = gb_clf_loaded.predict(X_new_scaled)
predictions_proba_rare = gb_clf_loaded.predict_proba(X_new_scaled)
predictions_proba = gb_clf_loaded.predict_proba(X_new_scaled)[:, 1]


print(predictions_proba_rare)

print("Predictions:", predictions)
print("Prediction Probabilities:", predictions_proba)
