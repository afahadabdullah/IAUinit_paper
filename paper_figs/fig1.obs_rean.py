import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import xarray as xr

# ==========================================
# File Paths
# ==========================================
# Paths for Reanalysis Initialization (ME0506) - 200505, 200506, 200507
me_paths = [
    '/nobackupp27/afahad/exp/IAU_exp/GEOSMIT_ME0506/holding/geosgcm_surf/200505/*surf*200505*z.nc4',
    '/nobackupp27/afahad/exp/IAU_exp/GEOSMIT_ME0506/holding/geosgcm_surf/200506/*surf*200506*z.nc4',
    '/nobackupp27/afahad/exp/IAU_exp/GEOSMIT_ME0506/holding/geosgcm_surf/200507/*surf*200507*z.nc4'
]

# IMERG data path
imerg_path = '/nobackupp27/afahad/project/IAUinit_paper/codes/data/imerge/*.nc4'

# ==========================================
# Data Loading and Processing
# ==========================================
print("Loading Reanalysis IC Data...")
start_date = '2005-05-05'
end_date = '2005-06-30'

try:
    ME506 = xr.open_mfdataset(me_paths)
    ME506P = ME506.PRECTOT.compute()
    # Convert from kg/m^2/s to mm/day
    ME506P = ME506P * 86400 

    # Slice for the analysis period aligned with precipitation spikes
    ME506P = ME506P.sel(time=slice(start_date, end_date))
    me_loaded = True
except Exception as e:
    print(f"Warning: Could not load Reanalysis IC data. Error: {e}")
    me_loaded = False

print("Loading IMERG Observation Data...")
try:
    # Use load to bring it into memory or keep as dask depending on size
    imerg_ds = xr.open_mfdataset(imerg_path)
    
    # IMERG usually has variable 'precipitationCal' in mm/hr
    var_name = [v for v in imerg_ds.data_vars if 'precip' in v.lower() or 'pr' in v.lower()][0]
    imerg_pr = imerg_ds[var_name].compute()
    
    # If the unit is mm/hr, multiply by 24 to get mm/day for equivalent comparison
    # (Assuming precipitationCal is in mm/hr)
    imerg_pr = imerg_pr * 24.0

    # Ensure time coordinates are sliced appropriately
    imerg_pr = imerg_pr.sel(time=slice(start_date, end_date))
    imerg_loaded = True
except Exception as e:
    print(f"Warning: Could not load IMERG data. Error: {e}")
    imerg_loaded = False
    imerg_pr = None

# ==========================================
# Extracting Data for Western Tropical Pacific
# ==========================================
# Region bounds from original precip_spike.ipynb
x1, x2 = 143, 143
y1, y2 = -1, -1

if me_loaded:
    # Extract regional mean for Reanalysis
    me_region = ME506P.sel(lat=slice(y1, y2), lon=slice(x1, x2)).mean(dim=['lon', 'lat'])

if imerg_loaded and imerg_pr is not None:
    # IMERG resolution is 0.1 degree. 
    # Use 'nearest' method for sub-setting exact coordinates in case of float mismatches.
    try:
        # Western Tropical Pacific coordinates: lat: -1, lon: 143
        # Depending on lon format (-180 to 180 or 0 to 360), adjust if needed
        # Assuming -180 to 180 standard for IMERG
        lon_target = 143 if imerg_pr.lon.max() > 180 else 143
        imerg_region = imerg_pr.sel(lat=-1, lon=lon_target, method='nearest')
        
        # Resample IMERG from half-hourly to 3-hourly or daily for cleaner plotting
        # (Optional, but plotting half-hourly might be very noisy)
        imerg_region = imerg_region.resample(time='3H').mean()
    except KeyError:
        print("Could not slice IMERG by lat/lon directly, you may need to adjust the coordinates.")
        imerg_region = None

# ==========================================
# Plotting
# ==========================================
if me_loaded:
    print("Generating Plot...")
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Plot Reanalysis IC
    ax.plot(me_region.time, me_region.values, color='blue', label='Reanalysis IC', linewidth=1.5)

    # Plot IMERG obs
    if imerg_loaded and imerg_region is not None:
        ax.plot(imerg_region.time, imerg_region.values, color='black', linestyle='-', 
                alpha=0.7, label='IMERG Obs (3-hourly mean)', linewidth=1.2)

    ax.set_title('Western Tropical Pacific Precipitation: Reanalysis IC vs IMERG Obs', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date (Month-Day)', fontsize=10)
    ax.set_ylabel('Precipitation (mm/day)', fontsize=10)
    
    # Same limits if desired:
    ax.set_ylim(-5, 135)
    
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Formatting the x-axis to Month-Day for clarity
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.xticks(rotation=45, fontsize=9)

    plt.tight_layout()
    output_filename = 'fig1.obs_rean.png'
    plt.savefig(output_filename, dpi=150)
    print(f"Plot saved successfully as {output_filename}")
else:
    print("Cannot generate plot without the Reanalysis data.")
