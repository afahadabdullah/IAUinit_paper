import os
# Disable HDF5 file locking to prevent filesystem errors on HPC
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
from tqdm import tqdm

# Parameters
data_dirs = [
    '/home6/afahad/nobackup27/exp/IAU_exp/GEOSMIT_ME05062/holding/geosgcm_surf/200505',
    '/home6/afahad/nobackup27/exp/IAU_exp/GEOSMIT_ME05062/holding/geosgcm_surf/200506',
    '/home6/afahad/nobackup27/exp/IAU_exp/GEOSMIT_ME05062/holding/geosgcm_surf/200507'
]
file_pattern = 'GEOSMIT_ME0506.geosgcm_surf.2*.nc4'
var_name = 'PRECTOT'
output_dir = '/home6/afahad/nobackup27/project/initialization_shock/paper_figs'

print("Starting analysis script...")

# Collect files
print("Searching for files...")
files = []
for d in data_dirs:
    fs = glob.glob(os.path.join(d, file_pattern))
    files.extend(fs)
files.sort()



if not files:
    print("No files found!")
    sys.exit(1)

print(f"Found {len(files)} files.")

# Load data
# Use chunks to handle large data, but for 2 months it might fit in memory if we subset early.
# 3-hourly data for 3 months = 90 days * 8 = 720 time steps.
# 720 * 144 * 91 (approx) * 8 bytes is small.

data_list = []

for i, f in tqdm(enumerate(files)):
    try:
        # 1. Open the single file
        # We use a context manager (with) to ensure it closes INSTANTLY after reading
        with xr.open_dataset(f) as ds_temp:
            
            # 2. Select ONLY the variable you need
            # This reduces memory usage and processing time
            if var_name in ds_temp:
                da_temp = ds_temp[var_name]
                
                # 3. Force load into memory (.load())
                # This reads the bits now and detaches from the file on disk.
                # It solves the "HDF5 error" and "stuck" issues.
                da_temp.load()
                
                data_list.append(da_temp)
            else:
                print(f"Warning: {var_name} not found in {os.path.basename(f)}")

        # Progress indicator
        if i % 20 == 0:
            print(f"Loaded {i}/{len(files)} files", flush=True)

    except Exception as e:
        print(f"Skipping corrupt file: {f}")
        print(f"Error: {e}")
        # Continue to the next file instead of crashing
        continue

print("Concatenating data...")

# 4. Combine into one DataArray
if not data_list:
    print("Error: No data loaded successfully.")
    sys.exit(1)

# Concatenate along the time dimension
da = xr.concat(data_list, dim='time')

# 5. Sort by time just in case files were out of order
da = da.sortby('time')

# Shift Longitude to 0-360
print("Shifting longitude to 0-360...")
da.coords['lon'] = (da.coords['lon'] + 360) % 360
da = da.sortby('lon')
print(f"New Lon Range: {da.lon.min().values} to {da.lon.max().values}")

print(f"Data loaded successfully. Shape: {da.shape}")#

# Select variable and fill NaNs if any
#da = ds[var_name]

# Region selection: Tropical strip -15 to 15
print(" selecting tropical band -15 to 15")
da_trop = da.sel(lat=slice(-16, 16)) # Include a bit more for symmetric filling? No, precise is better.
# Actually WK usually uses 15S-15N.

# To handle time, we should ensure monotonic increasing
# Convert PRECTOT to mm/day if needed. Usually units are kg/m2/s.
# 1 kg/m2/s = 86400 mm/day.
da_trop = da_trop * 86400.0
da_trop.attrs['units'] = 'mm/day'

# ----------------------------
# Hovmoller Diagram
# ----------------------------
print("Generating Hovmoller diagram...")
# Average over latitude
da_hov = da_trop.sel(lat=slice(-10, 10)).mean(dim='lat') # Use 10S-10N for cleaner signal? Or 15.
# Let's use 10S-10N as user mentioned Indo-Pacific to East Pacific.
da_hov = da_hov.compute(sync='continuous') 

plt.figure(figsize=(12, 10))
# Plot limited time range if needed? User said "available 2 months".
# Plot full time.
da_hov.plot(x='lon', y='time', cmap='Blues', robust=True)
plt.title(f'Hovmoller Diagram of {var_name} (10S-10N)')
plt.xlabel('Longitude')
plt.ylabel('Time')
plt.savefig(os.path.join(output_dir, 'hovmoller_precip.png'), dpi=150)
plt.close()
print("Hovmoller diagram saved.")

# ----------------------------
# Wheeler-Kiladis Diagram
# ----------------------------
print("Computing Wheeler-Kiladis Diagram...")

# 1. Preprocessing
# Remove time mean (and maybe seasonal cycle, but for 3 months, mean is 90%).
# Detrending?
anom = da_trop - da_trop.mean(dim='time')

# Split into Symmetric and Antisymmetric
# Interpolate to symmetric grid if lat is not symmetric?
# Lat is -90 to 90. 
# Check if lat is symmetric around 0.
lats = anom.lat.values
if not np.allclose(lats, -lats[::-1]):
    print("Warning: Latitudes not perfectly symmetric. Interpolating.")
    new_lat = np.linspace(-15, 15, 31) # 1 degree resolution
    anom = anom.interp(lat=new_lat)

# Get symmetric / antisymmetric
data_sym = 0.5 * (anom + anom.isel(lat=slice(None, None, -1)).values)
data_asym = 0.5 * (anom - anom.isel(lat=slice(None, None, -1)).values)

# Use Symmetric component for Kelvin waves (main focus usually)
# User mentioned "travel from west to east", likely Kelvin.
component = data_sym
component_name = "Symmetric"

# 2. FFT
# (time, lat, lon) -> (freq, lat, k)
# We want to sum power over latitude.
# Compute FFT over time and lon.
# Use windowing (tapering) to reduce spectral leakage?
# Standard WK uses tapering.
# Apply Hanning window in time.
# But simply: numpy fft.

# Load into memory
data_np = component.values # (time, lat, lon)
nt, nlat, nlon = data_np.shape

# Detrend in time
# data_np = scipy.signal.detrend(data_np, axis=0) # Only if scipy available.
# Simple linear detrend:
time_idx = np.arange(nt)
p = np.polyfit(time_idx, data_np.reshape(nt, -1), 1)
trend = p[0] * time_idx[:, None] + p[1]
data_np = data_np - trend.reshape(nt, nlat, nlon)

# Tapering
# window = np.hanning(nt)[:, None, None] * np.hanning(nlon)[None, None, :]
# data_np = data_np * window

# FFT
# axis 0 = time, axis 2 = lon (assuming lat is 1)
# Checking dims
# ds dims: (time, lat, lon) usually.
print(f"Data shape: {data_np.shape}")

fft_out = np.fft.fft2(data_np, axes=(0, 2))
power = np.abs(fft_out)**2

# Sum power over latitude
power_lat_sum = np.sum(power, axis=1) # (nt, nlon)

# Shift to center frequencies
power_shifted = np.fft.fftshift(power_lat_sum, axes=(0, 1))

# Frequencies
freqs = np.fft.fftshift(np.fft.fftfreq(nt, d=0.125)) # 3-hourly = 0.125 days
ks = np.fft.fftshift(np.fft.fftfreq(nlon, d=1.0/nlon)) # wavenumber

# Filter for plotting
# Positive frequencies only (CPD)
freq_mask = (freqs > 0) & (freqs < 0.8) # up to 0.8 cpd
k_mask = (ks >= -15) & (ks <= 15)

freq_idxs = np.where(freq_mask)[0]
k_idxs = np.where(k_mask)[0]

power_subset = power_shifted[freq_idxs[:, None], k_idxs]
freqs_sub = freqs[freq_idxs]
ks_sub = ks[k_idxs]

# Background smoothing (1-2-1 filter)
# Iterative smoothing
bg = power_subset.copy()
for _ in range(10): # Number of passes
    # Smooth in freq
    bg = 0.25 * np.roll(bg, 1, axis=0) + 0.5 * bg + 0.25 * np.roll(bg, -1, axis=0)
    # Smooth in k
    bg = 0.25 * np.roll(bg, 1, axis=1) + 0.5 * bg + 0.25 * np.roll(bg, -1, axis=1)

# Signal to noise
signal = np.log10(power_subset / bg)

# Plotting WK with Period
# Transform Freqs to Period (Days)
# Avoid division by zero (freqs > 0 already)
periods_sub = 1.0 / freqs_sub

plt.figure(figsize=(10, 8))
# Plotting vs Period (Y-axis)
# Use pcolormesh or contourf? Contourf is smoother.
# Note: Period is non-linear. The grid will be non-uniform in Period space.
# It's better to plot vs Freq but label as Period? or Plot vs Period directly?
# If plotting vs Period, the y-axis will be stretched at low freqs.
# Let's try plotting vs Period directly.
plt.contourf(ks_sub, periods_sub, signal, levels=20, cmap='Spectral_r')
plt.colorbar(label='Log10(Power/Background)')
plt.title(f'Wheeler-Kiladis Diagram ({component_name} PRECTOT)')
plt.xlabel('Zonal Wavenumber')
plt.ylabel('Period (Days)')
plt.grid(True, alpha=0.3)
plt.axvline(0, color='k', linestyle='--')
plt.ylim(0, 60) # Limit Period, though filtering might not cover it fully? Freqs > 0 => Period < Inf.
# Our min freq is 1/90 cpd (approx). Max Period 90. So 0-60 covers high freq part.

# Theoretical curves (Kelvin Wave)
# Period = 1/Freq = 1 / (c * k * factor)
# F ~ c * k
# T ~ 1 / (c * k)
Re = 6.371e6
for he, color in zip([12, 25, 50, 90], ['k', 'k', 'k', 'k']):
    c = np.sqrt(9.8 * he) # m/s
    kk = np.linspace(0.1, 15, 100) # Start from 0.1 to avoid div by zero
    ff = 86400.0 * c * kk / (2.0 * np.pi * Re) # cpd
    pp = 1.0 / ff # days
    plt.plot(kk, pp, color=color, linewidth=1.5, linestyle='-')
    
    # Label
    # Find index near Period=15
    idx_lbl = np.argmin(np.abs(pp - 15))
    if idx_lbl < len(kk):
        plt.text(kk[idx_lbl], pp[idx_lbl], f'{he}m', color='black', fontsize=10)


plt.savefig(os.path.join(output_dir, 'wk_diagram_period.png'), dpi=150)
plt.close()
print("WK diagram saved (Period space).")

# ----------------------------
# Spectral Filtering (Kelvin & MJO)
# ----------------------------
print("Performing Spectral Filtering...")

# Reuse logic from before but careful with variable names
# ... (Same Filtering Logic) ...

# Unshifted frequencies
freqs_un = np.fft.fftfreq(nt, d=0.125) 
ks_un = np.fft.fftfreq(nlon, d=1.0/nlon) 

fft_sym = np.fft.fft2(component.values, axes=(0, 2))

# MJO and Kelvin Masks
# Parameters
f_min_kel = 1/30.0; f_max_kel = 1/2.5; k_min_kel = 1; k_max_kel = 14
g = 9.8; Re = 6.371e6
h_min, h_max = 8, 90 
c_min = np.sqrt(g * h_min); c_max = np.sqrt(g * h_max)
factor = 86400.0 / (2.0 * np.pi * Re)

f_min_mjo = 1/96.0; f_max_mjo = 1/30.0; k_min_mjo = 1; k_max_mjo = 5

# Mask initialization
# mask_kel and mask_mjo need to match fft_sym shape: (nt, nlat, nlon)
mask_kel = np.zeros_like(fft_sym, dtype=bool)
mask_mjo = np.zeros_like(fft_sym, dtype=bool)

# Create 2D filter (Time, Lon) first
filt_kel_2d = np.zeros((nt, nlon), dtype=bool)
filt_mjo_2d = np.zeros((nt, nlon), dtype=bool)

for t in range(nt):
    f = freqs_un[t]
    for x in range(nlon):
        k = ks_un[x]
        
        # --- Kelvin ---
        if (f > f_min_kel) and (f < f_max_kel) and (k >= k_min_kel) and (k <= k_max_kel):
             f_calc = k * factor
             if (f >= c_min * f_calc) and (f <= c_max * f_calc):
                 filt_kel_2d[t, x] = True
        if (f < -f_min_kel) and (f > -f_max_kel) and (k <= -k_min_kel) and (k >= -k_max_kel):
             ratio = f / k
             c_eff = ratio / factor
             if (c_eff >= c_min) and (c_eff <= c_max):
                 filt_kel_2d[t, x] = True

        # --- MJO ---
        if (f > f_min_mjo) and (f < f_max_mjo) and (k >= k_min_mjo) and (k <= k_max_mjo):
            filt_mjo_2d[t, x] = True
        if (f < -f_min_mjo) and (f > -f_max_mjo) and (k <= -k_min_mjo) and (k >= -k_max_mjo):
            filt_mjo_2d[t, x] = True

# Broadcast to all latitudes
mask_kel[:] = filt_kel_2d[:, None, :]
mask_mjo[:] = filt_mjo_2d[:, None, :]

fft_kelvin = fft_sym * mask_kel
fft_mjo = fft_sym * mask_mjo

kelvin_data = np.fft.ifft2(fft_kelvin, axes=(0,2)).real
mjo_data = np.fft.ifft2(fft_mjo, axes=(0,2)).real

da_kelvin = xr.DataArray(kelvin_data, coords=component.coords, dims=component.dims, name='Kelvin_Precip')
da_mjo = xr.DataArray(mjo_data, coords=component.coords, dims=component.dims, name='MJO_Precip')

# ----------------------------
# Point Analysis
# ----------------------------
print("Analyzing Specific Grid Points...")

points = [
    {'label': 'Region 1 (143E, -1N)', 'lon': 143, 'lat': -1},
    {'label': 'Region 2 (215E, 8N)',  'lon': 215, 'lat': 8}, # 215 = -145 + 360
    {'label': 'Region 3 (233E, 7N)',  'lon': 233, 'lat': 7}  # 233 = -127 + 360
]

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

for i, pt in enumerate(points):
    ax = axes[i]
    # Select nearest point
    # da_trop for total anomaly (anom is anomaly)
    ts_anom = anom.sel(lon=pt['lon'], lat=pt['lat'], method='nearest')
    ts_kel = da_kelvin.sel(lon=pt['lon'], lat=pt['lat'], method='nearest')
    ts_mjo_pt = da_mjo.sel(lon=pt['lon'], lat=pt['lat'], method='nearest')
    
    ts_anom.plot(ax=ax, label='Precip Anomaly', color='black', alpha=0.6, linewidth=1.5)
    ts_kel.plot(ax=ax, label='Kelvin', color='red', linewidth=2.0)
    ts_mjo_pt.plot(ax=ax, label='MJO', color='blue', linewidth=2.0, linestyle='--')
    
    ax.set_title(f"{pt['label']} | Precip Decomposition")
    ax.set_ylabel('mm/day')
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(loc='upper right')

    # Quantitative Analysis for this point
    print(f"\n--- Analysis for {pt['label']} ---")
    # Find local maxima (spikes) or just sort by value? 
    # Let's simple pick top 5 distinct events.
    # To avoid picking adjacent points of same spike, we might need simple peak finding.
    # For now, let's just sort and pick top indices, then skip if close in time?
    # Or just top 5 values.
    
    # Sort descending
    sorted_indices = np.argsort(ts_anom.values)[::-1]
    top_indices = sorted_indices[:5]
    
    print(f"Top 5 Spikes:")
    print(f"{'Date':<20} | {'Total':<10} | {'Kelvin':<10} | {'MJO':<10} | {'% Kelvin':<10} | {'% MJO':<10}")
    print("-" * 85)
    
    for idx in top_indices:
        time_val = ts_anom.time.values[idx]
        val_tot = ts_anom.values[idx]
        val_kel = ts_kel.values[idx]
        val_mjo = ts_mjo_pt.values[idx]
        
        # Avoid division by zero, though spike should be large
        if abs(val_tot) < 0.1:
            pct_kel = 0.0
            pct_mjo = 0.0
        else:
            pct_kel = (val_kel / val_tot) * 100.0
            pct_mjo = (val_mjo / val_tot) * 100.0
            
        print(f"{str(time_val)[:19]:<20} | {val_tot:6.2f}     | {val_kel:6.2f}     | {val_mjo:6.2f}     | {pct_kel:6.1f}%     | {pct_mjo:6.1f}%")


plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'point_timeseries_decomposition.png'), dpi=150)
plt.close()
print("Point analysis plot saved.")
