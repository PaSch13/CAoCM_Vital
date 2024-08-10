import requests
import pandas as pd
from io import StringIO
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout



def getVitaDBData():
    # URL of the CSV file
    url = 'https://api.vitaldb.net/cases'

    # Send a GET request to fetch the CSV file
    response = requests.get(url)

    # Ensure the request was successful
    if response.status_code == 200:
        # Convert the response content to a pandas DataFrame
        df = pd.read_csv(StringIO(response.text))
        print("DataFrame loaded successfully:")
        print(df.head())  # Display the first few rows of the DataFrame
        print(df.columns)

        # Preprocessing:
        # Convert all string entries to lowercase
        df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        # Remove leading and trailing spaces from all string entries
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        return df

    else:
        print(f"Failed to fetch the CSV file. Status code: {response.status_code}")

        return None


def occurenceCounter(data, col):
    # Count the occurrences of each unique value in the "dx" column
    value_counts = data[col].value_counts()
    # Convert the result to a dictionary
    value_counts_dict = value_counts.to_dict()
    print(f"Value counts for column {col}:")
    print(value_counts_dict)

    return value_counts_dict

def sortBy(data, col):
    # Sort the DataFrame by the col column in descending order
    df_sorted = data.sort_values(by=col, ascending=False)
    
    print("\nDataFrame sorted by column 'xy' in descending order:")
    print(df_sorted[['optype', 'dx', 'opname', 'approach', col]])

    return df_sorted

#
#   Implementation of the NN
#
def create_nn_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)  # Output layer, no activation function (for regression)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    return model



df = getVitaDBData()

# Convert 'sex' column to numeric values: 'm' -> 1, 'f' -> 0
df['sex'] = df['sex'].map({'m': 1, 'f': 0})

# Convert 'optype' column to numeric values:
unique_values = df['optype'].unique()
mapping_dict = {value: index for index, value in enumerate(unique_values)}
df['optype'] = df['optype'].map(mapping_dict)

print(mapping_dict)

print(unique_values)

# Convert 'position' column to numeric values:
unique_values = df['position'].unique()
mapping_dict = {value: index for index, value in enumerate(unique_values)}
df['position'] = df['position'].map(mapping_dict)

# Remove rows with missing values in any of the selected columns
columns_to_check = ['age', 'sex', 'height', 'weight', 'bmi', 'asa', 'emop', 'optype']

df = df.dropna(subset=columns_to_check)
# Print the number of rows and columns in the cleaned dataset
print(f"Number of rows in the cleaned dataset: {df.shape[0]}")
print(f"Number of columns in the cleaned dataset: {df.shape[1]}")

# Define features (X) and target (y)
X = df[columns_to_check].values
y = df['icu_days'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (mean=0 and variance=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# Create the model
input_dim = X_train_scaled.shape[1]  # Number of features
model = create_nn_model(input_dim)

# Display model architecture
model.summary()


# Train the model
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_test_scaled, y_test), verbose=1)

# Evaluate the model
y_pred = model.predict(X_test_scaled).flatten()

for p in y_pred:
    if p > 10:
        print(p)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
mae = mean_absolute_error(y_test, np.abs(y_pred))

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")
print(f"Mean Absolute Error (MAE): {mae}")




import matplotlib.pyplot as plt

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Save the trained model
model.save('model_predictor/icu_predictor_model.h5')
