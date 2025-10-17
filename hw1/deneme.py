# Step Counting from Phone Accelerometer
# --------------------------------------
# Expected CSV headers for accelerometer: time, seconds_elapsed, x, y, z
# Optional gravity CSV: time, seconds_elapsed, x, y, z  (gravity sensor)
#
# What it does:
# 1) Load accel (+ optional gravity) CSVs
# 2) Align to uniform sampling using seconds_elapsed
# 3) Compute linear acceleration (either subtract gravity or high-pass)
# 4) Magnitude -> low-pass filter (0.3–5 Hz passband approx for walking/running)
# 5) Peak picking with cadence-aware constraints
#
# Outputs:
# - Printed step count
# - Plots of raw accel, linear accel, smoothed magnitude with detected peaks

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, find_peaks

# -------------------- USER CONFIG --------------------

ACCEL_CSV = "hw1/2025-10-12_16-18-03/Accelerometer.csv"      # your accelerometer file
GRAVITY_CSV = None             # e.g., "./gravity.csv" if you logged it; else leave None
TIME_COLS = dict(t="time", tsec="seconds_elapsed", x="x", y="y", z="z")

# If seconds_elapsed is missing, set a known sampling rate (Hz). Otherwise leave None.
FALLBACK_FS = None  # e.g., 50

# Filters (tune if needed)
HP_GRAVITY_CUTOFF = 0.3   # Hz, remove gravity via high-pass when no gravity sensor file
LP_SMOOTH_CUTOFF = 5.0    # Hz, keep gait band and suppress high-freq noise
NOTCH_HZ = None           # e.g., 50 or 60 to notch mains interference (optional)

# Peak detection (tune for your pace & sampling rate)
MIN_STEP_HZ = 0.6         # ~36 steps/min lower bound (very slow walk)
MAX_STEP_HZ = 4.0         # ~240 steps/min upper bound (sprint bound)
PEAK_PROMINENCE = 0.5     # m/s^2; raise if you get false positives
PEAK_HEIGHT = None        # m/s^2; optional absolute min height

# -----------------------------------------------------

def load_xyz(csv_path, cols=TIME_COLS):
    df = pd.read_csv(csv_path)
    for k in ["t","tsec","x","y","z"]:
        if cols[k] not in df.columns:
            raise ValueError(f"Column '{cols[k]}' not found in {csv_path}")
    tsec = df[cols["tsec"]].astype(float).values
    # normalize time to start at zero
    tsec = tsec - tsec[0]
    x = df[cols["x"]].astype(float).values
    y = df[cols["y"]].astype(float).values
    z = df[cols["z"]].astype(float).values
    return tsec, np.vstack([x,y,z]).T  # (N,), (N,3)

def estimate_fs(tsec, fallback=None):
    if len(tsec) < 3:
        return fallback or 50.0
    dt = np.median(np.diff(tsec))
    if dt <= 0 and fallback is None:
        raise ValueError("Non-increasing timestamps and no FALLBACK_FS given.")
    return (1.0/dt) if dt > 0 else float(fallback)

def butter_highpass(cut, fs, order=3):
    return butter(order, cut/(fs/2.0), btype="highpass")

def butter_lowpass(cut, fs, order=4):
    return butter(order, cut/(fs/2.0), btype="lowpass")

def apply_filter(b, a, x):
    # filtfilt for zero-phase
    if x.ndim == 1:
        return filtfilt(b, a, x)
    return np.column_stack([filtfilt(b, a, x[:,i]) for i in range(x.shape[1])])

def magnitude(xyz):
    return np.sqrt((xyz**2).sum(axis=1))

def compute_linear_accel(acc_xyz, fs, grav_xyz=None):
    if grav_xyz is not None:
        # direct subtraction if gravity sensor is available
        return acc_xyz - grav_xyz
    # else: high-pass remove gravity
    b, a = butter_highpass(HP_GRAVITY_CUTOFF, fs)
    return apply_filter(b, a, acc_xyz)

def optional_notch(sig, fs, f0):
    if f0 is None: return sig
    # Q ~ 30 is a mild notch; increase to narrow the notch
    b, a = iirnotch(w0=f0/(fs/2.0), Q=30)
    if sig.ndim == 1:
        return filtfilt(b, a, sig)
    return np.column_stack([filtfilt(b, a, sig[:,i]) for i in range(sig.shape[1])])

def detect_steps(smoothed_mag, fs):
    # convert cadence bounds to minimum distance in samples
    min_dist_samples = int(np.floor(fs / MAX_STEP_HZ))  # two peaks at most per 1/MAX_STEP_HZ seconds
    # find peaks
    peaks, props = find_peaks(
        smoothed_mag,
        distance=max(1, min_dist_samples),
        prominence=PEAK_PROMINENCE,
        height=PEAK_HEIGHT
    )
    # optional: enforce max step rate by pruning peaks that are too close
    if MIN_STEP_HZ is not None:
        min_period = 1.0 / MAX_STEP_HZ
        max_period = 1.0 / MIN_STEP_HZ
        t = np.arange(len(smoothed_mag))/fs
        kept = []
        last_t = -np.inf
        for p in peaks:
            tp = t[p]
            if tp - last_t < min_period:
                continue
            if kept and tp - last_t > max_period:
                # if a huge gap, we still accept; the distance filter already handled too-close peaks
                pass
            kept.append(p)
            last_t = tp
        peaks = np.array(kept, dtype=int)
    return peaks

# -------------------- Load data --------------------
t_acc, acc_xyz = load_xyz(ACCEL_CSV, TIME_COLS)
fs = estimate_fs(t_acc, FALLBACK_FS)

grav_xyz = None
if GRAVITY_CSV:
    t_g, grav_xyz = load_xyz(GRAVITY_CSV, TIME_COLS)
    # simple sync by length; for production, resample one to the other's timeline
    min_len = min(len(acc_xyz), len(grav_xyz))
    acc_xyz = acc_xyz[:min_len]
    grav_xyz = grav_xyz[:min_len]
    t_acc = t_acc[:min_len]

# -------------------- Pipeline --------------------
# 1) Compute linear acceleration
lin_xyz = compute_linear_accel(acc_xyz, fs, grav_xyz)

# optional mains notch on each axis
lin_xyz = optional_notch(lin_xyz, fs, NOTCH_HZ)

# 2) Magnitude (orientation-invariant)
lin_mag = magnitude(lin_xyz)

# 3) Low-pass filter to keep gait band
b_lp, a_lp = butter_lowpass(LP_SMOOTH_CUTOFF, fs, order=4)
lin_mag_smooth = filtfilt(b_lp, a_lp, lin_mag)

# 4) Peak detection
peaks = detect_steps(lin_mag_smooth, fs)
step_count = len(peaks)

print(f"Estimated steps: {step_count}")
if step_count > 1:
    stride_intervals = np.diff(peaks) / fs
    cadence_hz = np.mean(1.0/stride_intervals)
    cadence_spm = 60.0 * cadence_hz
    print(f"Average cadence: {cadence_spm:.1f} steps/min")

# -------------------- Plots --------------------
# Plot 1: raw accel magnitude (for intuition)
raw_mag = magnitude(acc_xyz)

plt.figure(figsize=(10,4))
plt.plot(t_acc, raw_mag, label="raw |acc|")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.title("Raw acceleration magnitude")
plt.tight_layout()
plt.show()

# Plot 2: linear accel magnitude (gravity removed)
plt.figure(figsize=(10,4))
plt.plot(t_acc, lin_mag, label="|linear acc|")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.title("Linear acceleration magnitude (gravity removed)")
plt.tight_layout()
plt.show()

# Plot 3: smoothed mag + detected steps
plt.figure(figsize=(10,4))
plt.plot(t_acc, lin_mag_smooth, label="smoothed |linear acc|")
plt.plot(t_acc[peaks], lin_mag_smooth[peaks], "x", label="steps")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.title(f"Detected steps: {step_count}")
plt.tight_layout()
plt.show()