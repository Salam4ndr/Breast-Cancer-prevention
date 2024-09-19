import pandas as pd

def load_data(file_path='data.csv'):
    """
    Load the dataset from a local file.

    Parameters:
    - file_path (str): Path to the CSV file. Default is 'data.csv'.

    Returns:
    - DataFrame: Pandas DataFrame containing the loaded dataset.
    """
    # Load the dataset from a local file path
    data = pd.read_csv(file_path)
    
    # Create a copy of the dataset for further analysis
    data_copy = data.copy()
    
    return data_copy
