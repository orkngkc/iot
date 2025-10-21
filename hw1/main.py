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

def custom_convolve(signal, kernel):
    """
    Custom implementation of convolution operation.
    
    Parameters:
    signal: Input signal array
    kernel: Convolution kernel array
    
    Returns:
    convolved: Convolved signal
    """
    signal_len = len(signal)
    kernel_len = len(kernel)
    output_len = signal_len + kernel_len - 1
    convolved = np.zeros(output_len)
    
    # Perform convolution
    for i in range(output_len):
        for j in range(kernel_len):
            if i - j >= 0 and i - j < signal_len:
                convolved[i] += signal[i - j] * kernel[j]
    
    return convolved

def custom_find_peaks(signal, height=None, distance=None):
    """
    Custom implementation of peak detection.
    
    Parameters:
    signal: Input signal array
    height: Minimum height for peaks (optional)
    distance: Minimum distance between peaks (optional)
    
    Returns:
    peaks: Array of peak indices
    properties: Dictionary with peak properties
    """
    peaks = []
    signal_len = len(signal)
    
    # Find all local maxima
    for i in range(1, signal_len - 1):
        # Check if current point is a local maximum
        if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            # Check height threshold if provided
            if height is None or signal[i] >= height:
                peaks.append(i)
    
    # Remove peaks that are too close together if distance is specified
    if distance is not None and len(peaks) > 0:
        filtered_peaks = [peaks[0]]  # Keep first peak
        for peak in peaks[1:]:
            if peak - filtered_peaks[-1] >= distance:
                filtered_peaks.append(peak)
        peaks = filtered_peaks
    
    # Get peak heights
    peak_heights = [signal[p] for p in peaks] if peaks else []
    
    properties = {
        'peak_heights': np.array(peak_heights)
    }
    
    return np.array(peaks), properties

accelerometer_data = load_sensor_data('2025-10-12_16-18-03/TotalAcceleration.csv')
gyroscope_data = load_sensor_data('2025-10-12_16-18-03/Gyroscope.csv')
gravity_data = load_sensor_data('2025-10-12_16-18-03/Gravity.csv')


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
    print("Gravity Removed")
    return lax, lay, laz

# Remove gravity using direct sensor data with proper timestamp alignment

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

# Plot magnitude with activity segments (using gravity-removed data)
plt.figure(figsize=(12,6))
gravity_removed_magnitude = np.sqrt(ax**2 + ay**2 + az**2)
plt.plot(acc['time'], gravity_removed_magnitude, color='black', linewidth=0.8)

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

t_acc = accelerometer_data['seconds_elapsed'].to_numpy(dtype=float)
# gravity data preparation
g = gravity_data[['seconds_elapsed','x','y','z']].to_numpy(dtype=float)
idx = np.argsort(g[:,0])
g_time = g[idx, 0]
gx = g[idx, 1]; gy = g[idx, 2]; gz = g[idx, 3]
lax, lay, laz = remove_gravity(ax, ay, az, t_acc, gx, gy, gz, g_time)
ax = lax
ay = lay
az = laz
# Compute basic features per segment using gravity-removed data
features = []
for (start, end, label) in segments:
    # Get time mask for this segment
    time_mask = (acc['time'] >= start) & (acc['time'] < end)
    
    # Calculate magnitude from gravity-removed data
    segment_ax = ax[time_mask]
    segment_ay = ay[time_mask] 
    segment_az = az[time_mask]
    segment_magnitude = np.sqrt(segment_ax**2 + segment_ay**2 + segment_az**2)
    
    mean_mag = segment_magnitude.mean()
    std_mag = segment_magnitude.std()
    features.append([label, mean_mag, std_mag])

features_df = pd.DataFrame(features, columns=["Activity", "Mean Magnitude", "Std Magnitude"])
print("\n=== Activity Summary Features (Gravity Removed) ===\n")
print(features_df)


# -------------------- PART 2: Step Detection --------------------





# order and remove duplicates
t_acc = np.sort(t_acc)
t_acc = t_acc[np.insert(np.diff(t_acc) > 0, 0, True)]  #remove duplicates and non-increasing timestamps

dt = np.diff(t_acc)
dt = dt[(dt > 0) & np.isfinite(dt)]
fs = 1.0 / np.median(dt)   







def sliding_windows(x, fs, win_sec=3.0, hop_sec=1.0):
    win = int(round(win_sec * fs))   # window size in samples
    hop = int(round(hop_sec * fs))   # hop size (how much we slide forward)
    for start in range(0, max(len(x) - win + 1, 0), hop):
        yield start, x[start:start+win]





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
        
        # Use our custom peak detection instead of find_peaks
        pks, _ = custom_find_peaks(seg, distance=min_dist_samples, height=thr)
        
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
# Only count steps during walking segment (120-180 seconds)
walking_mask = (t_acc >= 120) & (t_acc <= 180)
walking_ax = ax[walking_mask]
walking_ay = ay[walking_mask]
walking_az = az[walking_mask]
walking_t_acc = t_acc[walking_mask]

step_times = count_steps_streaming(walking_ax, walking_ay, walking_az, walking_t_acc, fs, win_sec=3.0, hop_sec=1.0)
step_count = len(step_times)

print(f"Estimated steps during walking (120-180s): {step_count}")
print(f"Walking duration: {180-120} seconds")
print(f"Average walking pace: {step_count/(180-120):.1f} steps/second = {step_count*60/(180-120):.1f} steps/minute")


# -------------------- PART 3: POSE ESTIMATION (EXTRA CREDIT) --------------------

# Load gyroscope data
gyro = gyroscope_data[['seconds_elapsed','x','y','z']].to_numpy(dtype=float)
idx = np.argsort(gyro[:,0])
gyro_time = gyro[idx, 0]
gx = gyro[idx, 1]; gy = gyro[idx, 2]; gz = gyro[idx, 3]

# Interpolate all sensor data to common timestamps
# Use accelerometer timestamps as reference
common_time = t_acc
ax_interp = np.interp(common_time, t_acc, ax)
ay_interp = np.interp(common_time, t_acc, ay)
az_interp = np.interp(common_time, t_acc, az)
gx_interp = np.interp(common_time, gyro_time, gx)
gy_interp = np.interp(common_time, gyro_time, gy)
gz_interp = np.interp(common_time, gyro_time, gz)

# Calculate sampling period
dt = np.mean(np.diff(common_time))

def complementary_filter(ax, ay, az, gx, gy, gz, dt, alpha=0.98):
    """
    Complementary filter for attitude estimation.
    
    Parameters:
    ax, ay, az: accelerometer readings (m/s²)
    gx, gy, gz: gyroscope readings (rad/s)
    dt: sampling period (s)
    alpha: filter coefficient (0-1, higher = more gyro trust)
    
    Returns:
    roll, pitch, yaw: estimated angles in radians
    """
    n = len(ax)
    roll = np.zeros(n)
    pitch = np.zeros(n)
    yaw = np.zeros(n)
    
    # Initial attitude from accelerometer (assuming no linear acceleration)
    roll[0] = np.arctan2(ay[0], np.sqrt(ax[0]**2 + az[0]**2))
    pitch[0] = np.arctan2(-ax[0], np.sqrt(ay[0]**2 + az[0]**2))
    yaw[0] = 0  # Cannot estimate yaw from accelerometer alone
    
    for i in range(1, n):
        # Gyroscope integration
        roll_gyro = roll[i-1] + gx[i] * dt
        pitch_gyro = pitch[i-1] + gy[i] * dt
        yaw_gyro = yaw[i-1] + gz[i] * dt
        
        # Accelerometer attitude estimation
        roll_acc = np.arctan2(ay[i], np.sqrt(ax[i]**2 + az[i]**2))
        pitch_acc = np.arctan2(-ax[i], np.sqrt(ay[i]**2 + az[i]**2))
        
        # Complementary filter
        roll[i] = alpha * roll_gyro + (1 - alpha) * roll_acc
        pitch[i] = alpha * pitch_gyro + (1 - alpha) * pitch_acc
        yaw[i] = yaw_gyro  # No accelerometer correction for yaw
    
    return roll, pitch, yaw

def madgwick_ahrs(ax, ay, az, gx, gy, gz, dt, beta=0.1):
    """
    Madgwick AHRS algorithm for attitude estimation.
    
    Parameters:
    ax, ay, az: accelerometer readings (m/s²)
    gx, gy, gz: gyroscope readings (rad/s)
    dt: sampling period (s)
    beta: algorithm gain
    
    Returns:
    roll, pitch, yaw: estimated angles in radians
    """
    n = len(ax)
    
    # Initialize quaternion
    q = np.zeros((n, 4))
    q[0] = [1, 0, 0, 0]  # w, x, y, z
    
    for i in range(1, n):
        # Normalize accelerometer measurement
        ax_norm = ax[i] / np.linalg.norm([ax[i], ay[i], az[i]])
        ay_norm = ay[i] / np.linalg.norm([ax[i], ay[i], az[i]])
        az_norm = az[i] / np.linalg.norm([ax[i], ay[i], az[i]])
        
        # Extract quaternion components
        qw, qx, qy, qz = q[i-1]
        
        # Gradient descent algorithm corrective step
        s1 = 2*qx*(2*qy*qy + 2*qz*qz - 1) - 2*qy*(2*qx*qy - 2*qw*qz) - 2*qz*(2*qx*qz + 2*qw*qy)
        s2 = 2*qy*(2*qx*qx + 2*qz*qz - 1) + 2*qx*(2*qx*qy - 2*qw*qz) - 2*qz*(2*qy*qz - 2*qw*qx)
        s3 = 2*qz*(2*qx*qx + 2*qy*qy - 1) - 2*qx*(2*qx*qz + 2*qw*qy) + 2*qy*(2*qy*qz - 2*qw*qx)
        s4 = 2*qw*(2*qx*qx + 2*qy*qy + 2*qz*qz - 1)
        
        # Normalize step magnitude
        s_norm = np.sqrt(s1**2 + s2**2 + s3**2 + s4**2)
        if s_norm != 0:
            s1 /= s_norm
            s2 /= s_norm
            s3 /= s_norm
            s4 /= s_norm
        
        # Compute rate of change of quaternion from gyroscope
        qDot1 = 0.5 * (-qx*gx[i] - qy*gy[i] - qz*gz[i])
        qDot2 = 0.5 * (qw*gx[i] + qy*gz[i] - qz*gy[i])
        qDot3 = 0.5 * (qw*gy[i] - qx*gz[i] + qz*gx[i])
        qDot4 = 0.5 * (qw*gz[i] + qx*gy[i] - qy*gx[i])
        
        # Apply feedback step
        qDot1 -= beta * s1
        qDot2 -= beta * s2
        qDot3 -= beta * s3
        qDot4 -= beta * s4
        
        # Integrate to yield quaternion
        qw += qDot1 * dt
        qx += qDot2 * dt
        qy += qDot3 * dt
        qz += qDot4 * dt
        
        # Normalize quaternion
        q_norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        q[i] = [qw/q_norm, qx/q_norm, qy/q_norm, qz/q_norm]
    
    # Convert quaternion to Euler angles
    roll = np.arctan2(2*(q[:,0]*q[:,1] + q[:,2]*q[:,3]), 1 - 2*(q[:,1]**2 + q[:,2]**2))
    pitch = np.arcsin(2*(q[:,0]*q[:,2] - q[:,3]*q[:,1]))
    yaw = np.arctan2(2*(q[:,0]*q[:,3] + q[:,1]*q[:,2]), 1 - 2*(q[:,2]**2 + q[:,3]**2))
    
    return roll, pitch, yaw

# Apply pose estimation algorithms
print("\n" + "="*60)
print("PART 3: POSE ESTIMATION")
print("="*60)
print("\nPose estimation determines the orientation of the device in 3D space.")
print("This is useful for understanding how the device was positioned and")
print("moved during different activities.")
print("\nWe calculate three rotation angles:")
print("• ROLL:  Rotation around X-axis (left-right tilting)")
print("• PITCH: Rotation around Y-axis (up-down tilting)")  
print("• YAW:   Rotation around Z-axis (left-right turning)")
print("\nTwo algorithms are implemented for comparison:")
print("• Complementary Filter: Simple fusion of accelerometer and gyroscope")
print("• Madgwick AHRS: Advanced quaternion-based algorithm")
print("\nThe following plots show the estimated orientation over time.")
print("="*60)

# Complementary filter
roll_comp, pitch_comp, yaw_comp = complementary_filter(ax_interp, ay_interp, az_interp, 
                                                       gx_interp, gy_interp, gz_interp, dt, alpha=0.98)

# Madgwick AHRS filter
roll_madgwick, pitch_madgwick, yaw_madgwick = madgwick_ahrs(ax_interp, ay_interp, az_interp,
                                                           gx_interp, gy_interp, gz_interp, dt, beta=0.1)

# Convert to degrees for visualization
roll_comp_deg = np.degrees(roll_comp)
pitch_comp_deg = np.degrees(pitch_comp)
yaw_comp_deg = np.degrees(yaw_comp)

roll_madgwick_deg = np.degrees(roll_madgwick)
pitch_madgwick_deg = np.degrees(pitch_madgwick)
yaw_madgwick_deg = np.degrees(yaw_madgwick)



fig, axes = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle('Device Orientation During Activities', fontsize=14, fontweight='bold')

# Roll angle plots
axes[0,0].plot(common_time, roll_comp_deg, 'b-', linewidth=1)
axes[0,0].set_title('Roll Angle - Complementary Filter')
axes[0,0].set_xlabel('Time (s)')
axes[0,0].set_ylabel('Roll (degrees)')
axes[0,0].grid(True)

axes[0,1].plot(common_time, roll_madgwick_deg, 'r-', linewidth=1)
axes[0,1].set_title('Roll Angle - Madgwick AHRS')
axes[0,1].set_xlabel('Time (s)')
axes[0,1].set_ylabel('Roll (degrees)')
axes[0,1].grid(True)

# Pitch angle plots
axes[1,0].plot(common_time, pitch_comp_deg, 'b-', linewidth=1)
axes[1,0].set_title('Pitch Angle - Complementary Filter')
axes[1,0].set_xlabel('Time (s)')
axes[1,0].set_ylabel('Pitch (degrees)')
axes[1,0].grid(True)

axes[1,1].plot(common_time, pitch_madgwick_deg, 'r-', linewidth=1)
axes[1,1].set_title('Pitch Angle - Madgwick AHRS')
axes[1,1].set_xlabel('Time (s)')
axes[1,1].set_ylabel('Pitch (degrees)')
axes[1,1].grid(True)

# Yaw angle plots
axes[2,0].plot(common_time, yaw_comp_deg, 'b-', linewidth=1)
axes[2,0].set_title('Yaw Angle - Complementary Filter')
axes[2,0].set_xlabel('Time (s)')
axes[2,0].set_ylabel('Yaw (degrees)')
axes[2,0].grid(True)

axes[2,1].plot(common_time, yaw_madgwick_deg, 'r-', linewidth=1)
axes[2,1].set_title('Yaw Angle - Madgwick AHRS')
axes[2,1].set_xlabel('Time (s)')
axes[2,1].set_ylabel('Yaw (degrees)')
axes[2,1].grid(True)

# Add activity segments
colors = ['#b0bec5', '#c5e1a5', '#81d4fa', '#ef9a9a']
segments = [(0, 60, 'Sitting'), (60, 120, 'Standing'), (120, 180, 'Walking'), (180, 240, 'Running')]

for i, (start, end, label) in enumerate(segments):
    for ax in axes.flat:
        ax.axvspan(start, end, color=colors[i], alpha=0.1, zorder=0)

plt.tight_layout()
plt.show()


# Compare methods side by side
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(common_time, roll_comp_deg, 'b-', label='Complementary Filter', linewidth=1)
axes[0].plot(common_time, roll_madgwick_deg, 'r-', label='Madgwick AHRS', linewidth=1)
axes[0].set_title('Roll Angle Comparison')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Roll (degrees)')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(common_time, pitch_comp_deg, 'b-', label='Complementary Filter', linewidth=1)
axes[1].plot(common_time, pitch_madgwick_deg, 'r-', label='Madgwick AHRS', linewidth=1)
axes[1].set_title('Pitch Angle Comparison')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Pitch (degrees)')
axes[1].legend()
axes[1].grid(True)

axes[2].plot(common_time, yaw_comp_deg, 'b-', label='Complementary Filter', linewidth=1)
axes[2].plot(common_time, yaw_madgwick_deg, 'r-', label='Madgwick AHRS', linewidth=1)
axes[2].set_title('Yaw Angle Comparison')
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Yaw (degrees)')
axes[2].legend()
axes[2].grid(True)

# Add activity segments
colors = ['#b0bec5', '#c5e1a5', '#81d4fa', '#ef9a9a']
segments = [(0, 60, 'Sitting'), (60, 120, 'Standing'), (120, 180, 'Walking'), (180, 240, 'Running')]

for i, (start, end, label) in enumerate(segments):
    for ax in axes:
        ax.axvspan(start, end, color=colors[i], alpha=0.1, zorder=0)

plt.tight_layout()
plt.show()

# Print summary statistics
print("=== POSE ESTIMATION SUMMARY ===")
print(f"Data duration: {common_time[-1] - common_time[0]:.1f} seconds")
print(f"Sampling rate: {1/dt:.1f} Hz")
print(f"Total samples: {len(common_time)}")

print("\n--- Complementary Filter Results ---")
print(f"Roll range: {roll_comp_deg.min():.1f}° to {roll_comp_deg.max():.1f}°")
print(f"Pitch range: {pitch_comp_deg.min():.1f}° to {pitch_comp_deg.max():.1f}°")
print(f"Yaw range: {yaw_comp_deg.min():.1f}° to {yaw_comp_deg.max():.1f}°")

print("\n--- Madgwick AHRS Results ---")
print(f"Roll range: {roll_madgwick_deg.min():.1f}° to {roll_madgwick_deg.max():.1f}°")
print(f"Pitch range: {pitch_madgwick_deg.min():.1f}° to {pitch_madgwick_deg.max():.1f}°")
print(f"Yaw range: {yaw_madgwick_deg.min():.1f}° to {yaw_madgwick_deg.max():.1f}°")

print("\n=== POSE ESTIMATION COMPLETED ===")



