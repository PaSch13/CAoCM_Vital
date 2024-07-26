import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the model
model = tf.keras.models.load_model('CaoCom_Model_StaticData/model_predictor/icu_predictor_model.h5')

columns_to_check = ['age', 'sex', 'height', 'weight', 'bmi', 'asa', 'emop', 'optype']

x = input("...")

dataframe_newData = 5 # ?!?!!?

# @Patrick bitte Werte umwandeln:
# 1) Mann: 1, Frau: 0
# 2) Optype Dictionary: {'colorectal': 0, 'stomach': 1, 'biliary/pancreas': 2, 'vascular': 3, 'major resection': 4, 'breast': 5, 'minor resection': 6, 'transplantation': 7, 'hepatic': 8, 'thyroid': 9, 'others': 10}

# Preprocess the new data
scaler = StandardScaler()
df_new_scaled = scaler.fit_transform(dataframe_newData[columns_to_check])

# Make predictions with the loaded model
predictions = model.predict(df_new_scaled).flatten()

print(f"Predictions: {predictions}")
