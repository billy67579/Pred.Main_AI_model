import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer

# Load your dataset
file_path = "/home/besong/project/project_data/turbine_1/turb_1.csv"  # Your actual file path
data = pd.read_csv(file_path)

# List of all sensor columns to process
sensors = [col for col in data.columns if col.startswith('Sensor')]

# Function to apply scaling methods and plot distributions
def plot_distributions(sensor_data, sensor_name):
    """
    Apply MinMaxScaler, StandardScaler, and PowerTransformer (Yeo-Johnson) to the data
    and plot the distributions.

    Parameters:
        sensor_data (pd.Series): Data for the specific sensor.
        sensor_name (str): Name of the sensor (e.g., 'Sensor_0').
    """
    # Drop NaN values and convert to NumPy array
    variable_data = sensor_data.dropna().values

    # Create a DataFrame for analysis
    df = pd.DataFrame({'Original': variable_data})

    # Apply MinMaxScaler
    min_max_scaler = MinMaxScaler()
    df['MinMaxScaler'] = min_max_scaler.fit_transform(df[['Original']])

    # Apply StandardScaler
    standard_scaler = StandardScaler()
    df['StandardScaler'] = standard_scaler.fit_transform(df[['Original']])

    # Apply PowerTransformer (Yeo-Johnson)
    power_transformer = PowerTransformer(method='yeo-johnson')
    df['PowerTransformer_YeoJohnson'] = power_transformer.fit_transform(df[['Original']])

    # Plot the distributions
    plt.figure(figsize=(16, 12))  # Increase figure size

    # Original data distribution
    plt.subplot(2, 2, 1)
    sns.histplot(df['Original'], kde=True, bins=30, color='blue', alpha=0.6)
    plt.title(f'{sensor_name} - Original Data Distribution', fontsize=14, pad=20)  # Add padding to title
    plt.xlabel('Value', fontsize=12, labelpad=10)  # Add padding to x-axis label
    plt.ylabel('Frequency', fontsize=12)

    # MinMaxScaler distribution
    plt.subplot(2, 2, 2)
    sns.histplot(df['MinMaxScaler'], kde=True, bins=30, color='green', alpha=0.6)
    plt.title(f'{sensor_name} - MinMaxScaler Distribution', fontsize=14, pad=20)  # Add padding to title
    plt.xlabel('MinMaxSCaler_Value (0 to 1)', fontsize=12, labelpad=10)  # Add padding to x-axis label
    plt.ylabel('Frequency', fontsize=12)

    # StandardScaler distribution
    plt.subplot(2, 2, 3)
    sns.histplot(df['StandardScaler'], kde=True, bins=30, color='orange', alpha=0.6)
    plt.title(f'{sensor_name} - StandardScaler Distribution', fontsize=14, pad=20)  # Add padding to title
    plt.xlabel('Standardized Value (mean=0, std=1)', fontsize=12, labelpad=10)  # Add padding to x-axis label
    plt.ylabel('Frequency', fontsize=12)

    # PowerTransformer (Yeo-Johnson) distribution
    plt.subplot(2, 2, 4)
    sns.histplot(df['PowerTransformer_YeoJohnson'], kde=True, bins=30, color='purple', alpha=0.6)
    plt.title(f'{sensor_name} - PowerTransformer (Yeo-Johnson) Distribution', fontsize=14, pad=20)  # Add padding to title
    plt.xlabel('Transformed Value', fontsize=12, labelpad=10)  # Add padding to x-axis label
    plt.ylabel('Frequency', fontsize=12)

    # Adjust layout and spacing
    plt.tight_layout(pad=5.0)  # Add more padding between subplots
    plt.subplots_adjust(top=0.92)  # Add more space at the top for titles

    plt.show()


# Loop through each sensor and plot distributions
for sensor in sensors:
    print(f"Processing {sensor}...")
    plot_distributions(data[sensor], sensor)
