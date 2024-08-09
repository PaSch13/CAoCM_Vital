import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pickle


# Function to create sequences
def create_sequences(df, sequence_length, existing_columns):
    sequences = []
    targets = []
    for caseid in df["caseid"].unique():
        case_data = df[df["caseid"] == caseid]
        for i in range(len(case_data) - sequence_length):
            seq = case_data.iloc[i : i + sequence_length][existing_columns[:-2]].values
            target = case_data.iloc[i + sequence_length]["icu_days"]
            sequences.append(seq)
            targets.append(target)
    return np.array(sequences), np.array(targets)


# Load existing columns
existing_columns = np.load("existing_columns.npy")

# Initialize the model outside the loop to keep training the same model
sequence_length = 150  # Adjust this value as needed

# Create the initial LSTM model
model = Sequential()
model.add(tf.keras.layers.Input(shape=(sequence_length, len(existing_columns) - 2)))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")

# Number of iterations for generating new data and retraining
num_iterations = 5

for iteration in range(num_iterations):
    print(f"--- Iteration {iteration + 1} ---")

    # Load preprocessed data
    full_data = pd.read_csv("preprocessed_data.csv")

    # Select cases with icu_days > 1
    data_icu_greater1 = full_data[full_data["icu_days"] > 1]
    caseids = data_icu_greater1["caseid"].unique()
    np.random.seed(42 + iteration)  # Change seed in each iteration
    np.random.shuffle(caseids)
    caseids = caseids[:23]
    data_icu_greater1 = data_icu_greater1[data_icu_greater1["caseid"].isin(caseids)]

    # Select cases with icu_days <= 1
    data_icu_smaller1 = full_data[full_data["icu_days"] <= 1]
    caseids = data_icu_smaller1["caseid"].unique()
    np.random.seed(42 + iteration)  # Change seed in each iteration
    np.random.shuffle(caseids)
    caseids = caseids[:23]
    data_icu_smaller1 = data_icu_smaller1[data_icu_smaller1["caseid"].isin(caseids)]

    # Combine the two datasets
    data_loaded = pd.concat([data_icu_greater1, data_icu_smaller1])
    print("Data loaded")

    # Standardize the data
    scaler = StandardScaler()
    data_loaded[existing_columns[:-2]] = scaler.fit_transform(data_loaded[existing_columns[:-2]])

    # Create sequences
    X, y = create_sequences(data_loaded, sequence_length, existing_columns)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Verify shapes
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Continue training the existing model with the new data
    history = model.fit(X_train, y_train, epochs=12, batch_size=32, validation_split=0.2)

    # Evaluate the model after training
    loss = model.evaluate(X_test, y_test, verbose=2)
    print(f"Mean Squared Error after iteration {iteration + 1}: {loss}")

    # Save the model after each iteration
    model.save(f"my_model_iteration_{iteration}.keras")
    model.save(f"my_model.keras")

    # Save the training history
    with open(f"history_dict_iteration_{iteration}.pkl", "wb") as file:
        pickle.dump(history.history, file)
    print(f"Completed iteration {iteration + 1}\n")


# save history
import pickle

with open("history_dict", "wb") as file:
    pickle.dump(history.history, file)


# Plot training & validation loss values
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.show()

# Predict on the test set
y_pred = model.predict(X_test)

# Plot the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test, label="Actual ICU Days")
plt.plot(y_pred, label="Predicted ICU Days")
plt.title("Actual vs Predicted ICU Days")
plt.xlabel("Sample")
plt.ylabel("ICU Days")
plt.legend(loc="upper right")
plt.show()
