import pandas as pd
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load existing columns
existing_columns = np.load("existing_columns.npy")

# Load preprocessed data
full_data = pd.read_csv("preprocessed_data.csv")

# only take data with icu_days > 1
data_icu_greater1 = full_data[full_data["icu_days"] > 1]


# take random 40 caseids
caseids = data_icu_greater1["caseid"].unique()
np.random.seed(42)
np.random.shuffle(caseids)
caseids = caseids[:23]
data_icu_greater1 = data_icu_greater1[data_icu_greater1["caseid"].isin(caseids)]

# only take data with icu_days <= 1
data_icu_smaller1 = full_data[full_data["icu_days"] <= 1]
# take random 40 caseids
caseids = data_icu_smaller1["caseid"].unique()
np.random.seed(42)
np.random.shuffle(caseids)
caseids = caseids[:23]
data_icu_smaller1 = data_icu_smaller1[data_icu_smaller1["caseid"].isin(caseids)]


# Combine the two datasets
data_loaded = pd.concat([data_icu_greater1, data_icu_smaller1])
print("data_loaded")
# Standardize the data
scaler = StandardScaler()
data_loaded[existing_columns[:-2]] = scaler.fit_transform(data_loaded[existing_columns[:-2]])


# Create sequences function
def create_sequences(df, sequence_length):
    sequences = []
    targets = []
    count = 0
    for caseid in df["caseid"].unique():
        case_data = df[df["caseid"] == caseid]
        count += 1
        print(count, "/", len(df["caseid"].unique()), len(case_data) - sequence_length, end="\r")
        for i in range(len(case_data) - sequence_length):
            seq = case_data.iloc[i : i + sequence_length][existing_columns[:-2]].values
            target = case_data.iloc[i + sequence_length]["icu_days"]
            sequences.append(seq)
            targets.append(target)
    return np.array(sequences), np.array(targets)


# # Create sequences
sequence_length = 150  # Adjust this value as needed
X, y = create_sequences(data_loaded, sequence_length)

# Save the sequences and targets
np.save("X.npy", X)
np.save("y.npy", y)
print("Saved sequences and targets")

# Load the sequences and targets
X = np.load("X.npy")
y = np.load("y.npy")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verify shapes
print(
    f"X_train shape: {X_train.shape}"
)  # Expected shape: (number of samples, sequence_length, number of features)
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")  # Expected shape: (number of samples,)
print(f"y_test shape: {y_test.shape}")


# Check the number of features
num_features = X_train.shape[2]
print(f"Number of features: {num_features}")


# Define the LSTM model
model = Sequential()
model.add(
    tf.keras.layers.Input(shape=(sequence_length, num_features))
)  # Use Input layer to define input shape
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")
# Evaluate the model and repeat this multiple times for batches
loss = model.evaluate(X_test, y_test, verbose=2)
print(f"Untrained Mean Squared Error: {loss}")
# Train the model
history = model.fit(X_train, y_train, epochs=12, batch_size=32, validation_split=0.2)

# Evaluate the model
loss = model.evaluate(X_test, y_test, verbose=2)
print(f"Mean Squared Error: {loss}")
# Save the model
model.save("my_model.keras")

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
