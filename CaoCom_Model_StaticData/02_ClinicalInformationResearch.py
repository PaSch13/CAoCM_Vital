import requests
import pandas as pd
from io import StringIO

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

data = getVitaDBData()

col_occurence = 'optype'
occurence = occurenceCounter(data, col_occurence)
# Interesting cols: optype, dx

# Store to Excel
df_occurence = pd.DataFrame(occurence, index=[0])
df_occurence = (df_occurence.T)
df_occurence.to_excel(f"occurence_{col_occurence}.xlsx")

exit

col_sort = 'icu_days'
sorted = sortBy(data, col_sort)
sorted.to_excel(f"sorted_{col_sort}.xlsx")

