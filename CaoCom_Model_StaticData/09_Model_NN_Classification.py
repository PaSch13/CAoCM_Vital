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


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Define and compile the model
def create_classification_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model




df = getVitaDBData()

# Convert 'sex' column to numeric values: 'm' -> 1, 'f' -> 0
df['sex'] = df['sex'].map({'m': 1, 'f': 0})

# Convert 'opname' column to numeric values:
unique_values = df['opname'].unique()
mapping_dict = {value: index for index, value in enumerate(unique_values)}
df['opname'] = df['opname'].map(mapping_dict)

# Convert 'dx' column to numeric values:
unique_values = df['dx'].unique()
mapping_dict = {value: index for index, value in enumerate(unique_values)}
df['dx'] = df['dx'].map(mapping_dict)

# Convert 'optype' column to numeric values:
unique_values = df['optype'].unique()
mapping_dict = {value: index for index, value in enumerate(unique_values)}
df['optype'] = df['optype'].map(mapping_dict)

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

# Transform target variable: 0 if ICU_Days <= 1, 1 if ICU_Days > 1
df['ICU_Class'] = np.where(df['icu_days'] <= 1, 0, 1)

# Inspect the class distribution
print(df['ICU_Class'].value_counts())
# Separate the majority and minority classes
df_majority = df[df['ICU_Class'] == 0]
df_minority = df[df['ICU_Class'] == 1]
# Undersample the majority class
df_majority_undersampled = df_majority.sample(len(df_minority), random_state=42)
# Combine the undersampled majority class with the minority class
df_balanced = pd.concat([df_majority_undersampled, df_minority])
# Shuffle the resulting DataFrame
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
# Inspect the class distribution to confirm balancing
print(df_balanced['ICU_Class'].value_counts())

df = df_balanced

# Define features (X) and new target (y)
X = df[['age', 'sex', 'height', 'weight', 'bmi', 'asa', 'emop']].values
y = df['ICU_Class'].values

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (mean=0 and variance=1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Create the model
input_dim = X_train_scaled.shape[1]  # Number of features
model = create_classification_model(input_dim)

# Display model architecture
model.summary()


# Early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, 
                    validation_data=(X_test_scaled, y_test), callbacks=[early_stopping], verbose=1)




# Evaluate the model
y_pred_proba = model.predict(X_test_scaled).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
