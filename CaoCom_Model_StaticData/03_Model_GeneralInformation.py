import requests
import pandas as pd
from io import StringIO
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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
#   Implementation of the different models
#
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train_linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {'model': model, 'mse': mse, 'r2': r2}

from sklearn.linear_model import Ridge

def train_ridge_regression(X_train, X_test, y_train, y_test):
    model = Ridge()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {'model': model, 'mse': mse, 'r2': r2}

from sklearn.linear_model import Lasso

def train_lasso_regression(X_train, X_test, y_train, y_test):
    model = Lasso()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {'model': model, 'mse': mse, 'r2': r2}

from sklearn.linear_model import ElasticNet

def train_elasticnet_regression(X_train, X_test, y_train, y_test):
    model = ElasticNet()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {'model': model, 'mse': mse, 'r2': r2}

from sklearn.tree import DecisionTreeRegressor

def train_decision_tree_regression(X_train, X_test, y_train, y_test):
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {'model': model, 'mse': mse, 'r2': r2}

from sklearn.ensemble import RandomForestRegressor

def train_random_forest_regression(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {'model': model, 'mse': mse, 'r2': r2}

from xgboost import XGBRegressor

def train_gradient_boosting_regression(X_train, X_test, y_train, y_test):
    model = XGBRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {'model': model, 'mse': mse, 'r2': r2}

from sklearn.svm import SVR

def train_support_vector_regression(X_train, X_test, y_train, y_test):
    model = SVR()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {'model': model, 'mse': mse, 'r2': r2}


df = getVitaDBData()


# Convert 'sex' column to numeric values: 'm' -> 1, 'f' -> 0
df['sex'] = df['sex'].map({'m': 1, 'f': 0})

# Remove rows with missing values in any of the selected columns
columns_to_check = ['age', 'sex', 'height', 'weight', 'bmi', 'asa', 'emop']
df_clean = df.dropna(subset=columns_to_check)

# Print the number of rows and columns in the cleaned dataset
print(f"Number of rows in the cleaned dataset: {df_clean.shape[0]}")
print(f"Number of columns in the cleaned dataset: {df_clean.shape[1]}")

# Save the cleaned dataframe to an Excel file
df_clean.to_excel("cleaned_data.xlsx", index=False)

# Define features and target
X = df_clean[['age', 'sex', 'height', 'weight', 'bmi', 'asa', 'emop']]
y = df_clean['icu_days']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


models_to_train = [
    train_linear_regression,
    train_ridge_regression,
    train_lasso_regression,
    train_elasticnet_regression,
    train_decision_tree_regression,
    train_random_forest_regression,
    train_gradient_boosting_regression,
    train_support_vector_regression
]

results = {}

for model_fn in models_to_train:
    model_name = model_fn.__name__[6:]  # Get model name without 'train_'
    print(f"Training {model_name}...")
    model_results = model_fn(X_train, X_test, y_train, y_test)
    results[model_name] = model_results

# Print results
for model_name, result in results.items():
    print(f"Model: {model_name}")
    print(f"MSE: {result['mse']}")
    print(f"R-squared: {result['r2']}")
    print("\n")


