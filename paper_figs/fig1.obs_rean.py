import os
import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import xarray as xr

# ==========================================
# File Paths
# ==========================================
# Paths for Reanalysis Initialization (ME0506) - only 200505 needed for May 5-9
me_paths = [
    '/nobackupp27/afahad/exp/IAU_exp/GEOSMIT_ME0506/holding/geosgcm_surf/200505/*surf*200505*z.nc4',
]

# IMERG data path
imerg_path = '/nobackupp27/afahad/project/IAUinit_paper/codes/data/imerge/*.nc4'

# ==========================================
# Data Loading and Processing
# ==========================================
print("Loading Reanalysis IC Data...")
start_date = '2005-05-06'
end_date = '2005-05-31'

try:
    print(f"Opening Reanalysis files from: {me_paths[0]} ...")
    # Expand glob patterns explicitly
    me_files = sorted([f for p in me_paths for f in glob.glob(p)])
    print(f"Found {len(me_files)} Reanalysis files.")
    ME506 = xr.open_mfdataset(me_files, engine='netcdf4', combine='by_coords', parallel=False)
    print("Computing Reanalysis precipitation...")
    ME506P = ME506.PRECTOT.compute(scheduler='synchronous')
    # Convert from kg/m^2/s to mm/day
    print("Converting precipitation units from kg/m^2/s to mm/day...")
    ME506P = ME506P * 86400 

    # Slice for the analysis period aligned with precipitation spikes
    print(f"Slicing Reanalysis data for time period: {start_date} to {end_date}...")
    ME506P = ME506P.sel(time=slice(start_date, end_date))
    me_loaded = True
    print("Reanalysis data loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load Reanalysis IC data. Error: {e}")
    me_loaded = False

print("\nLoading IMERG Observation Data...")
try:
    print(f"Opening IMERG files from: {imerg_path} (Parallel=False, Engine=netcdf4) ...")
    # Force the netcdf4 engine to prevent HDF5 backend attribute locks
    # Expand glob pattern, then filter to only files within the date range
    all_imerg_files = sorted(glob.glob(imerg_path))
    # IMERG filenames contain the date as YYYYMMDD, e.g. ...3IMERG.20050505-S...
    start_yyyymmdd = start_date.replace('-', '')  # '20050505'
    end_yyyymmdd   = end_date.replace('-', '')    # '20050509'
    imerg_files = [
        f for f in all_imerg_files
        if any(
            yyyymmdd >= start_yyyymmdd and yyyymmdd <= end_yyyymmdd
            for yyyymmdd in [os.path.basename(f).split('.')[4].split('-')[0]]
        )
    ]
    print(f"Found {len(imerg_files)} IMERG files in date range ({start_date} to {end_date}).")
    imerg_ds = xr.open_mfdataset(imerg_files, engine='netcdf4', combine='by_coords', parallel=False)
    
    # IMERG usually has variable 'precipitationCal' in mm/hr
    var_name = [v for v in imerg_ds.data_vars if 'precip' in v.lower() or 'pr' in v.lower()][0]
    print(f"Identified IMERG precipitation variable as: {var_name}")
    print("Computing IMERG precipitation...")
    imerg_pr = imerg_ds[var_name].compute(scheduler='synchronous')
    
    # If the unit is mm/hr, multiply by 24 to get mm/day for equivalent comparison
    # (Assuming precipitationCal is in mm/hr)
    print("Scaling IMERG precipitation from mm/hr to mm/day...")
    imerg_pr = imerg_pr * 24.0

    # Ensure time coordinates are sliced appropriately
    print(f"Slicing IMERG data for time period: {start_date} to {end_date}...")
    imerg_pr = imerg_pr.sel(time=slice(start_date, end_date))
    
    # Downscale IMERG to match Reanalysis grid cell using true mass-conserving approach:
    # Instead of coarsening the entire field (which may have grid offset issues),
    # we slice the native 0.1-deg IMERG data to the exact bounds of the target
    # Reanalysis 1-deg grid cell and take the area mean.
    # For Reanalysis grid cell centered at (lat=-1, lon=143), the cell spans:
    #   lat: -1.5 to -0.5,  lon: 142.5 to 143.5
    imerg_loaded = True
    print("IMERG data loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load or process IMERG data. Error: {e}")
    imerg_loaded = False
    imerg_pr = None

# ==========================================
# Extracting Data for Western Tropical Pacific
# ==========================================
print("\nExtracting data for the Western Tropical Pacific region...")
# Region bounds from original precip_spike.ipynb
# Reanalysis grid cell center
lat_center = -1
lon_center = 143

if me_loaded:
    print(f"Extracting Reanalysis data at lat: {lat_center}, lon: {lon_center}...")
    # Extract nearest grid cell for Reanalysis
    me_region = ME506P.sel(lat=lat_center, lon=lon_center, method='nearest')
    print("Reanalysis point data extracted.")

imerg_region = None
if imerg_loaded and imerg_pr is not None:
    try:
        # Mass-conserving: average all 0.1-deg IMERG cells within the 1-deg Reanalysis cell
        # Cell bounds: center ± 0.5 degrees
        lat_lo, lat_hi = lat_center - 0.5, lat_center + 0.5
        lon_lo, lon_hi = lon_center - 0.5, lon_center + 0.5
        print(f"Mass-conserving average of IMERG within cell bounds:")
        print(f"  lat: [{lat_lo}, {lat_hi}], lon: [{lon_lo}, {lon_hi}]")
        
        # Slice IMERG to the cell bounds and take area mean
        imerg_cell = imerg_pr.sel(
            lat=slice(lat_lo, lat_hi),
            lon=slice(lon_lo, lon_hi)
        ).mean(dim=['lat', 'lon'])
        
        # Resample IMERG from half-hourly to 3-hourly for cleaner plotting
        print("Resampling IMERG data to 3-hourly means for plotting...")
        imerg_region = imerg_cell.resample(time='3H').mean()
        print("IMERG regional data extracted and resampled.")
    except Exception as e:
        print(f"Could not extract IMERG for target cell. Error: {e}")
        imerg_region = None

# ==========================================
# Plotting
# ==========================================
if me_loaded:
    print("\nGenerating 2-Panel Plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # --- Panel (a): Comparison ---
    print("Plotting Panel (a): Comparison...")
    ax1.plot(me_region.time, me_region.values, color='blue', label='Reanalysis IC', linewidth=1.5)
    
    if imerg_loaded and imerg_region is not None:
        ax1.plot(imerg_region.time, imerg_region.values, color='black', linestyle='-', 
                alpha=0.7, label='IMERG Obs (3-hourly mean)', linewidth=1.2)

    ax1.set_title('(a) Western Tropical Pacific Precipitation: Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Precipitation (mm/day)', fontsize=10)
    ax1.set_ylim(-5, 135)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Panel (b): Difference (Reanalysis - IMERG) ---
    if imerg_loaded and imerg_region is not None:
        print("Plotting Panel (b): Difference (Reanalysis IC - IMERG)...")
        # Align time and compute difference
        # Interpolate IMERG to match Reanalysis time points exactly since 
        # timestamps might be slightly offset between datasets
        try:
            imerg_aligned = imerg_region.interp(time=me_region.time, method='nearest')
            diff = me_region - imerg_aligned
            
            # Plot the difference
            ax2.plot(me_region.time, diff.values, color='red', label='Difference (Rean-IMERG)', linewidth=1.5)
        except Exception as e:
            print(f"Error computing difference: {e}")
            ax2.text(0.5, 0.5, 'Error computing difference', ha='center', va='center')

        ax2.axhline(0, color='black', linewidth=1, linestyle='--')
        ax2.set_title('(b) Precipitation Difference: Reanalysis IC minus IMERG', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Difference (mm/day)', fontsize=10)
        ax2.legend(fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.6)
    else:
        ax2.text(0.5, 0.5, 'IMERG data not available for difference', ha='center', va='center')

    # Formatting the x-axis to Month-Day for clarity
    print("Configuring plot aesthetics (titles, labels, legends)...")
    ax2.set_xlabel('Date (Month-Day)', fontsize=10)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    plt.xticks(rotation=45, fontsize=9)

    plt.tight_layout()
    output_filename = 'fig1.obs_rean.png'
    print(f"Saving plot to {output_filename}...")
    plt.savefig(output_filename, dpi=150)
    print(f"Plot saved successfully as {output_filename}")
else:
    print("\nCannot generate plot without the Reanalysis data.")
