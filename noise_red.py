import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
import pywt
import matplotlib.pyplot as plt

# Define noise reduction techniques
def low_pass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def band_pass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def wavelet_denoising(data, wavelet='db4', level=3):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    threshold = np.median(np.abs(coeffs[-1])) / 0.6745 * np.sqrt(2 * np.log(len(data)))
    denoised_coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(denoised_coeffs, wavelet)

# Load raw vibration data from CSV file
raw_data = pd.read_csv("/home/besong/project/project_data/turbine_1/turb_1.csv")  # Replace with your file path

# Print statistics for each sensor
print("Raw Data Statistics:")
for sensor in raw_data.columns:
    if sensor != "Time_second":
        print(f"\n{sensor} Statistics:")
        print(raw_data[sensor].describe())

# Plot Power Spectral Density (PSD) for all sensors
plt.figure(figsize=(12, 6))
for sensor in raw_data.columns:
    if sensor != "Time_second":
        f, Pxx = welch(raw_data[sensor], fs=1000)  # Example sampling frequency
        plt.semilogy(f, Pxx, label=sensor)
plt.title("Power Spectral Density (PSD) - Noise Analysis")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.legend()
plt.grid()
plt.show()

# Normalize raw data for visualization
print("\nNormalizing Raw Data for Visualization...")
for sensor in raw_data.columns:
    if sensor != "Time_second":
        raw_data[sensor] = (raw_data[sensor] - raw_data[sensor].mean()) / raw_data[sensor].std()

# Parameters for filtering
sampling_frequency = 1000  # Example: 1000 Hz
low_pass_cutoff = 50  # Low-pass filter cutoff frequency (e.g., 50 Hz)
band_pass_lowcut = 10  # Band-pass filter low cutoff (e.g., 10 Hz)
band_pass_highcut = 200  # Band-pass filter high cutoff (e.g., 200 Hz)

# Initialize a new DataFrame to store processed data
processed_data = pd.DataFrame()
processed_data["Time_second"] = raw_data["Time_second"]  # Retain the time column

# Process each sensor column
for sensor in raw_data.columns:
    if sensor == "Time_second":
        continue  # Skip the time column

    print(f"Processing column: {repr(sensor)}")  # Use repr() to debug column names

    # Check if the column contains valid numeric data
    if not np.issubdtype(raw_data[sensor].dtype, np.number):
        print(f"{sensor} is not numeric. Skipping...")
        continue

    # Extract raw data for the sensor
    sensor_data = raw_data[sensor].values

    # Handle missing values (e.g., fill or interpolate)
    if np.any(pd.isnull(sensor_data)):
        print(f"Missing values detected in {sensor}. Filling with interpolation.")
        sensor_data = pd.Series(sensor_data).interpolate().fillna(0).values

    try:
        # Apply low-pass filter
        low_passed = low_pass_filter(sensor_data, low_pass_cutoff, sampling_frequency)

        # Apply band-pass filter
        band_passed = band_pass_filter(sensor_data, band_pass_lowcut, band_pass_highcut, sampling_frequency)

        # Apply wavelet denoising
        denoised = wavelet_denoising(sensor_data)

        # Add processed data to the new DataFrame
        processed_data[f"{sensor}_Low_Pass"] = low_passed
        processed_data[f"{sensor}_Band_Pass"] = band_passed
        processed_data[f"{sensor}_Wavelet_Denoised"] = denoised

    except Exception as e:
        print(f"Error processing {sensor}: {e}")
        continue

# Save the processed data to a new CSV file
processed_data.to_csv("processed_turb_1.csv", index=False)

# Cross-correlation analysis
print("\nCross-Correlation Analysis:")
for i in range(len(raw_data.columns) - 2):  # Exclude "Time_second"
    for j in range(i + 1, len(raw_data.columns) - 1):
        sensor1 = raw_data.columns[i + 1]
        sensor2 = raw_data.columns[j + 1]
        correlation = np.corrcoef(raw_data[sensor1], raw_data[sensor2])[0, 1]
        print(f"Correlation between {sensor1} and {sensor2}: {correlation:.3f}")

# Plot the processed data for verification
for sensor in raw_data.columns:
    if sensor == "Time_second":
        continue

    if f"{sensor}_Low_Pass" not in processed_data.columns:
        print(f"No processed data for {sensor}. Skipping plot.")
        continue

    plt.figure(figsize=(12, 6))
    plt.plot(raw_data["Time_second"], raw_data[sensor], label="Normalized Raw Data", alpha=0.7)
    plt.plot(raw_data["Time_second"], processed_data[f"{sensor}_Low_Pass"], label="Low-Pass Filter", alpha=0.7)
    plt.plot(raw_data["Time_second"], processed_data[f"{sensor}_Band_Pass"], label="Band-Pass Filter", alpha=0.7)
    plt.plot(raw_data["Time_second"], processed_data[f"{sensor}_Wavelet_Denoised"], label="Wavelet Denoised", alpha=0.7)
    plt.title(f"Sensor: {sensor}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend(loc="center right")
    plt.grid()
    plt.show()

print("Noise reduction complete. Processed data saved to 'processed_turb_1.csv'.")
