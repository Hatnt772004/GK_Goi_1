def load_data(file_path):
    """
    Load data from a CSV file.
    
    Parameters:
    file_path (str): The path to the CSV file.
    
    Returns:
    DataFrame: A pandas DataFrame containing the loaded data.
    """
    import pandas as pd
    return pd.read_csv(file_path)

def summarize_data(df):
    """
    Summarize the data by providing basic statistics.
    
    Parameters:
    df (DataFrame): The DataFrame to summarize.
    
    Returns:
    DataFrame: A DataFrame containing summary statistics.
    """
    return df.describe()

def visualize_data(df, column):
    """
    Visualize the distribution of a specified column in the DataFrame.
    
    Parameters:
    df (DataFrame): The DataFrame containing the data.
    column (str): The column name to visualize.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], bins=30, kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()