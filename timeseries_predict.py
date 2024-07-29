import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow.keras.models import load_model


def predict_icu_days_from_data(patient_data) -> float:
    # import icu_model.h5 and history_dict
    existing_columns = np.load("existing_columns.npy")

    model = load_model("lstm_model.keras")
    # drop first and alst 40 rows
    patient_data = patient_data.iloc[200:-40]

    patient_data = patient_data.fillna(method="ffill").fillna(method="bfill")
    # compare columns with existing_columns
    # missing_columns = [col for col in patient_data.columns if col not in existing_columns]
    # print(f"Missing columns: {missing_columns}")
    # drop time
    patient_data = patient_data.drop(columns=["time"])

    # Standardize the data
    scaler = StandardScaler()
    patient_data_scaled = scaler.fit_transform(patient_data)

    # Convert to DataFrame to maintain column names
    patient_data_scaled = pd.DataFrame(patient_data_scaled, columns=existing_columns[:-2])

    print("Patient data loaded and standardized.")

    # Create sequences function for a single patient
    def create_patient_sequence(df, sequence_length):
        sequences = []
        if len(df) >= sequence_length:
            for i in range(len(df) - sequence_length):
                seq = df.iloc[i : i + sequence_length][existing_columns[:-2]].values
                sequences.append(seq)
        return np.array(sequences)

    # Set the sequence length
    sequence_length = 1000  # Use the same value as used during training

    # Create the sequence
    X_patient = create_patient_sequence(patient_data_scaled, sequence_length)

    print(f"Patient sequence created: X_patient shape: {X_patient.shape}")

    # Ensure that the sequence length and number of features match the model's expected input shape
    if X_patient.shape[1:] == (sequence_length, len(existing_columns) - 2):
        # Make predictions
        y_pred_patient = model.predict(X_patient)
        print(y_pred_patient)
        # Average the predictions if multiple sequences were created
        predicted_icu_days = np.mean(y_pred_patient)
        return predicted_icu_days
        print(f"Predicted ICU days for the patient: {predicted_icu_days}")
    else:
        print("Error: The input sequence does not match the expected shape for the model.")
        return None
