import joblib
import numpy as np

# Load the model and scaler
gb_clf_loaded = joblib.load('CaoCom_Model_StaticData/model_classifier/gradient_boosting_model.pkl')
scaler_loaded = joblib.load('CaoCom_Model_StaticData/model_classifier/scaler.pkl')

# Function to preprocess new data
def preprocess_new_data(data):
    # @Patrick: Wie auch immer du es einpflegst: Männlich hat die 1 und Weiblich die 0 als Input für das Modell
    X_new_scaled = scaler_loaded.transform(data)
    return X_new_scaled

new_data = None # ?!?!?

# Preprocess and predict
X_new_scaled = preprocess_new_data(new_data)
predictions = gb_clf_loaded.predict(X_new_scaled)
predictions_proba = gb_clf_loaded.predict_proba(X_new_scaled)[:, 1]

print("Predictions:", predictions)
print("Prediction Probabilities:", predictions_proba)
