import pandas as pd

def process_data(file_path='data.csv'):
    """
    Process the dataset by loading, cleaning, and analyzing it.

    Parameters:
    - file_path (str): Path to the CSV file. Default is 'data.csv'.

    Returns:
    - DataFrame: Cleaned pandas DataFrame.
    """
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Display the first few rows of the dataset
    print("First 5 rows of the dataset:")
    print(data.head())
    
    # Display the last few rows of the dataset
    print("\nLast 5 rows of the dataset:")
    print(data.tail())
    
    # Display information about the dataset
    print("\nDataset information:")
    print(data.info())
    
    # Drop the 'Unnamed: 32' column and the 'id' column
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    
    # Convert the 'diagnosis' column to binary values
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})
    
    # Save the cleaned DataFrame to a variable
    data_save = data
    
    # Print the first 5 rows of the cleaned DataFrame
    print("\nFirst 5 rows of the cleaned dataset:")
    print(data_save.head())
    
    # Check for missing values in the dataset
    def num_missing(x):
        return sum(x.isnull())
    
    print("\nMissing values per column:")
    print(data_save.apply(num_missing, axis=0))
    
    # Print descriptive statistics of the dataset
    print("\nDescriptive statistics of the dataset:")
    print(data_save.describe().T)
    
    return data_save