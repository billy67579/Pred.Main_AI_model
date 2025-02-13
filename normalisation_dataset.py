import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset (replace with your file path)
file_path = "/home/besong/project/project_data/turbine_1/turb_1.csv"  # Update with your actual file path
raw_data = pd.read_csv(file_path)

# Function to normalize the dataset using vectorized operations for efficiency
def normalize_dataset(data, exclude_columns=None):
    """
    Normalize the dataset to scale values between 0 and 1, excluding specific columns.
    
    Parameters:
        data (pd.DataFrame): Input dataset.
        exclude_columns (list): List of columns to exclude from normalization (e.g., time column).
        
    Returns:
        pd.DataFrame: Normalized dataset.
    """
    if exclude_columns is None:
        exclude_columns = []
    
    normalized_data = data.copy()
    for column in data.columns:
        if column not in exclude_columns:  # Skip excluded columns
            col_min = data[column].min()
            col_max = data[column].max()
            normalized_data[column] = (data[column] - col_min) / (col_max - col_min)
    
    return normalized_data

# Normalize the dataset
normalized_data = normalize_dataset(raw_data, exclude_columns=["Time_second"])

# Save the normalized dataset to a new CSV file
normalized_file_path = "/home/besong/project/project_data/turbine_1/normalized_turb_1.csv"  # Update with your desired save location
normalized_data.to_csv(normalized_file_path, index=False)
print(f"Normalized dataset saved to {normalized_file_path}")

# Set Seaborn style to make the plot visually appealing
sns.set(style="whitegrid", palette="muted", font_scale=1.2)

# Generate separate plots for each sensor
for column in normalized_data.columns:
    if column != "Time_second":  # Skip the time column
        plt.figure(figsize=(10, 6))  # Create a new figure for each sensor
        sns.histplot(
            normalized_data[column],
            kde=True,  # Enable kernel density estimation (density curve)
            bins=50,   # Adjust number of bins for better visualization
            stat="density",  # Normalize histogram to show probability density
            alpha=0.4,  # Set transparency for histogram
        )
        plt.title(f"Normalized Data for {column}", fontsize=16)
        plt.xlabel("Normalized Values (0 to 1)", fontsize=14)
        plt.ylabel("Density", fontsize=14)
        plt.grid(visible=True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()
