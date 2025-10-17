import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def load_sensor_data(file_path):
    """
    Load sensor data from a CSV file into a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file containing sensor data.

    Returns:
    pd.DataFrame: A DataFrame containing the sensor data.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

accelerometer_data = load_sensor_data('hw1/2025-10-12_16-18-03/Accelerometer.csv')
gyroscope_data = load_sensor_data('hw1/2025-10-12_16-18-03/Gyroscope.csv')
gravity_data = load_sensor_data('hw1/2025-10-12_16-18-03/Gravity.csv')

# --- PART 1: Visualization and Feature Analysis ---

# Prepare accelerometer data
acc = accelerometer_data.copy()
acc.columns = [c.strip().lower() for c in acc.columns]
acc['time'] = acc['seconds_elapsed']

# Compute magnitude
acc['magnitude'] = np.sqrt(acc['x']**2 + acc['y']**2 + acc['z']**2)

# Plot magnitude with activity segments
plt.figure(figsize=(12,6))
plt.plot(acc['time'], acc['magnitude'], color='black', linewidth=0.8)

segments = [
    (0, 60, 'Standing'),
    (60, 120, 'Sitting'),
    (120, 180, 'Walking'),
    (180, 240, 'Running')
]
colors = ['#b0bec5', '#c5e1a5', '#81d4fa', '#ef9a9a']

for i, (start, end, label) in enumerate(segments):
    plt.axvspan(start, end, color=colors[i], alpha=0.3, label=label)

plt.title("Accelerometer Magnitude over Time")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration Magnitude (m/s²)")
plt.legend()
plt.tight_layout()
plt.show()

# Compute basic features per segment
features = []
for (start, end, label) in segments:
    segment = acc[(acc['time'] >= start) & (acc['time'] < end)]
    mean_mag = segment['magnitude'].mean()
    std_mag = segment['magnitude'].std()
    features.append([label, mean_mag, std_mag])

features_df = pd.DataFrame(features, columns=["Activity", "Mean Magnitude", "Std Magnitude"])
print("\n=== Activity Summary Features ===\n")
print(features_df)