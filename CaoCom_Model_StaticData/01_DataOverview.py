import requests
import pandas as pd
from io import StringIO

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

else:
    print(f"Failed to fetch the CSV file. Status code: {response.status_code}")
