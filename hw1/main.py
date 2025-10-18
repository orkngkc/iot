import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks


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

accelerometer_data = load_sensor_data('hw1/2025-10-12_16-18-03/TotalAcceleration.csv')
gyroscope_data = load_sensor_data('hw1/2025-10-12_16-18-03/Gyroscope.csv')
gravity_data = load_sensor_data('hw1/2025-10-12_16-18-03/Gravity.csv')

# --- PART 1: Visualization and Feature Analysis ---

# Prepare accelerometer data
acc = accelerometer_data.copy()
acc.columns = [c.strip().lower() for c in acc.columns]
acc['time'] = acc['seconds_elapsed']

# Compute magnitude
acc['magnitude'] = np.sqrt(acc['x']**2 + acc['y']**2 + acc['z']**2)

ax = acc['x'].values
ay = acc['y'].values
az = acc['z'].values

# fs değeri hesaplanmalı

# Plot magnitude with activity segments
plt.figure(figsize=(12,6))
plt.plot(acc['time'], acc['magnitude'], color='black', linewidth=0.8)

segments = [
    (0, 60, 'Sitting'),
    (60, 120, 'Standing'),
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


# -------------------- PART 2: Step Detection --------------------



t_acc = accelerometer_data['seconds_elapsed'].to_numpy(dtype=float)

# order and remove duplicates
t_acc = np.sort(t_acc)
t_acc = t_acc[np.insert(np.diff(t_acc) > 0, 0, True)]  #remove duplicates and non-increasing timestamps

dt = np.diff(t_acc)
dt = dt[(dt > 0) & np.isfinite(dt)]
fs = 1.0 / np.median(dt)   

# gravity data preparation
g = gravity_data[['seconds_elapsed','x','y','z']].to_numpy(dtype=float)
idx = np.argsort(g[:,0])
g_time = g[idx, 0]
gx = g[idx, 1]; gy = g[idx, 2]; gz = g[idx, 3]





def sliding_windows(x, fs, win_sec=3.0, hop_sec=1.0):
    win = int(round(win_sec * fs))   # window size in samples
    hop = int(round(hop_sec * fs))   # hop size (how much we slide forward)
    for start in range(0, max(len(x) - win + 1, 0), hop):
        yield start, x[start:start+win]

def remove_gravity(ax, ay, az, acc_times, gravity_x, gravity_y, gravity_z, gravity_times):
    """
    Remove gravity from accelerometer data using direct sensor readings with timestamp alignment.
    
    Parameters:
    ax, ay, az: accelerometer data arrays
    acc_times: accelerometer timestamps
    gravity_x, gravity_y, gravity_z: gravity sensor data arrays
    gravity_times: gravity sensor timestamps
    
    Returns:
    lax, lay, laz: linear acceleration (gravity removed)
    """
    # Interpolate gravity data to match accelerometer timestamps
    gravity_x_interp = np.interp(acc_times, gravity_times, gravity_x)
    gravity_y_interp = np.interp(acc_times, gravity_times, gravity_y)
    gravity_z_interp = np.interp(acc_times, gravity_times, gravity_z)
    
    # Remove interpolated gravity from accelerometer data
    lax = ax - gravity_x_interp
    lay = ay - gravity_y_interp
    laz = az - gravity_z_interp
    return lax, lay, laz

# Remove gravity using direct sensor data with proper timestamp alignment
lax, lay, laz = remove_gravity(ax, ay, az, t_acc, gx, gy, gz, g_time)



def magnitude(ax, ay, az):
    return np.sqrt(ax*ax + ay*ay + az*az)


# calculate lowpass filter but with time domain since it is time series data instead of converting frequency domain and multiplying
# we can use convolution with the kernel
def fir_lowpass_kernel(fc_hz, fs, num_taps=101):
    fc = fc_hz / fs
    M = num_taps - 1
    n = np.arange(num_taps)
    h = np.sinc(2*fc*(n - M/2))
    w = 0.54 - 0.46*np.cos(2*np.pi*n/M)   # Hamming window
    h *= w
    h /= np.sum(h)
    return h

def lowpass_fir(x, fs, fc_hz=3.0, num_taps=101):
    h = fir_lowpass_kernel(fc_hz, fs, num_taps)
    pad = len(h)//2
    xpad = np.pad(x, (pad, pad), mode='reflect')
    ypad = np.convolve(xpad, h, mode='same')
    return ypad[pad:-pad]

# helper for peak detection threshold
def robust_threshold(x, k=1.0):
    # median + k * MAD (median absolute deviation); MAD ~ robust spread
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return med + k * mad


def count_steps_streaming(ax, ay, az, t, fs, win_sec=3.0, hop_sec=1.0):
    mag = magnitude(ax, ay, az)
    mag_f = lowpass_fir(mag, fs, fc_hz=3.0, num_taps=101)

    step_times = []  # collect timestamps of accepted peaks
    min_dist_samples = int(fs / 3.0)  # ~max 3 Hz cadence

    for start, seg in sliding_windows(mag_f, fs, win_sec=win_sec, hop_sec=hop_sec):
        seg_t = t[start:start+len(seg)]   # artık t dışarıdan geldi
        thr = robust_threshold(seg, k=1.0)
        pks, _ = find_peaks(seg, distance=min_dist_samples, height=thr)
        for pk in pks:
            step_times.append(seg_t[pk])

    step_times = np.array(sorted(step_times))
    min_dt = 1.0 / 3.0  # seconds (≈ 3 Hz)
    dedup = [step_times[0]] if len(step_times) else []
    for s in step_times[1:]:
        if s - dedup[-1] >= min_dt:
            dedup.append(s)
    return np.array(dedup)

# ax, ay, az from accelerometer data take them from file
step_times = count_steps_streaming(ax, ay, az, t_acc, fs, win_sec=3.0, hop_sec=1.0)
step_count = len(step_times)

print(f"Estimated steps: {step_count}")


