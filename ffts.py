import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, convolve
import matplotlib.pyplot as plt

# Load the dataset (replace with your file path)
file_path = "/home/besong/project/project_data/turb_1.csv"  # Update with your actual file path
data = pd.read_csv(file_path)

# Sampling rate (in Hz) - calculated from the `Time_second` column
time_intervals = data["Time_second"].diff().mean()  # Average time step between rows
sampling_rate = 1 / time_intervals
print(f"Sampling rate: {sampling_rate:.2f} Hz")

# FFT Function
def perform_fft(sensor_data, sampling_rate):
    n = len(sensor_data)  # Number of samples
    fft_result = np.fft.fft(sensor_data)  # Perform FFT
    fft_amplitudes = np.abs(fft_result) / n  # Normalize FFT amplitude
    fft_amplitudes = fft_amplitudes[:n // 2]  # Take only the positive half of frequencies
    freqs = np.fft.fftfreq(n, d=1/sampling_rate)[:n // 2]  # Calculate frequency bins
    return freqs, fft_amplitudes

# Smoothing function (moving average)
def smooth_fft(fft_amplitudes, window_size=5):
    return convolve(fft_amplitudes, np.ones(window_size)/window_size, mode='same')

# Clean and preprocess sensor data
def preprocess_sensor_data(sensor_data):
    """
    Clean and preprocess the sensor data.
    - Replace NaN values with the column mean.
    - Subtract the mean to remove the DC component.

    Returns:
        Preprocessed sensor data.
    """
    # Replace NaN values with the column mean
    if np.any(np.isnan(sensor_data)):
        print("NaN values found in sensor data. Replacing with column mean.")
        sensor_data = np.nan_to_num(sensor_data, nan=np.nanmean(sensor_data))
    
    # Subtract the mean to remove the DC component
    sensor_data = sensor_data - np.mean(sensor_data)
    return sensor_data

# Debugging: Check raw and preprocessed data
for sensor in data.columns:
    if sensor != "Time_second":  # Skip the `Time_second` column
        sensor_data = data[sensor].values  # Extract the sensor data

        # Preprocess sensor data
        sensor_data = preprocess_sensor_data(sensor_data)

        # Perform FFT
        freqs, amplitudes = perform_fft(sensor_data, sampling_rate)

        # Apply smoothing to the FFT result (optional)
        amplitudes_smoothed = smooth_fft(amplitudes, window_size=10)

        # Correct frequency range selection
        if sensor in ["sensor_0", "sensor_1"]:
            max_frequency = 10  # Limit frequency range to 20 Hz for sensor_0 and sensor_1
            freq_interval = 1  # X-axis ticks every 1 Hz
        else:
            max_frequency = 150  # Limit frequency range to 200 Hz for other sensors
            freq_interval = 10  # X-axis ticks every 10 Hz

        # Ensure we filter out only the frequencies up to max_frequency
        freq_mask = (freqs >= 0) & (freqs <= max_frequency)
        freqs = freqs[freq_mask]
        amplitudes_smoothed = amplitudes_smoothed[freq_mask]

        # Skip invalid data
        if len(freqs) == 0 or len(amplitudes_smoothed) == 0 or np.max(amplitudes_smoothed) == 0:
            print(f"Skipping plot for {sensor} due to empty or zero amplitude data.")
            continue

        # Plot the frequency spectrum with linear scaling for amplitude
        plt.figure(figsize=(8, 4))
        plt.plot(freqs, amplitudes_smoothed, color='black', linewidth=0.3)  # Linear scale for amplitude
        plt.title(f"FFT Frequency Spectrum for {sensor} (Linear Scale)", fontsize=14)
        plt.xlabel("Frequency (Hz)", fontsize=12)
        plt.ylabel("Amplitude", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)

        # Adjust frequency intervals on the x-axis
        plt.xticks(np.arange(0, max_frequency + 1, freq_interval))  # Dynamic interval based on sensor type
        plt.xlim(0, max_frequency)  # Ensure the x-axis goes from 0 to max_frequency
        
        plt.tight_layout()
        plt.show()
