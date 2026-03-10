import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point

# ==========================================
# File Paths and Cache Configuration
# ==========================================
# Paths for Reanalysis Initialization (ME0506) - only 200505 needed for May 5-9
me_paths = [
    '/nobackupp27/afahad/exp/IAU_exp/GEOSMIT_ME0506/holding/geosgcm_surf/200505/*surf*200505*z.nc4',
]

# IMERG data path
imerg_path = '/nobackupp27/afahad/project/IAUinit_paper/codes/data/imerge/*.nc4'

# Cache Directory
cache_dir = 'data'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
    print(f"Created cache directory: {cache_dir}")

me_cache_file = os.path.join(cache_dir, 'me_mean_200505.nc4')
imerge_cache_file = os.path.join(cache_dir, 'imerge_mean_200505.nc4')

# ==========================================
# Data Loading and Processing
# ==========================================
print("Loading Reanalysis IC Data...")
start_date = '2005-05-06'
end_date = '2005-05-31'

# --- Process Reanalysis ---
if os.path.exists(me_cache_file):
    print(f"Loading Reanalysis mean from cache: {me_cache_file}")
    me_mean = xr.open_dataarray(me_cache_file)
    me_loaded = True
else:
    try:
        print(f"Opening Reanalysis files from: {me_paths[0]} ...")
        me_files = sorted([f for p in me_paths for f in glob.glob(p)])
        print(f"Found {len(me_files)} Reanalysis files.")
        ME506 = xr.open_mfdataset(me_files, engine='netcdf4', combine='by_coords', parallel=False)
        print("Computing Reanalysis precipitation...")
        ME506P = ME506.PRECTOT.compute(scheduler='synchronous')
        
        # Convert from kg/m^2/s to mm/day
        print("Converting precipitation units from kg/m^2/s to mm/day...")
        ME506P = ME506P * 86400 

        # Slice for the analysis period
        print(f"Slicing Reanalysis data for time period: {start_date} to {end_date}...")
        ME506P = ME506P.sel(time=slice(start_date, end_date))
        
        # Time mean
        me_mean = ME506P.mean(dim='time')
        
        print(f"Saving Reanalysis mean to cache: {me_cache_file}")
        me_mean.to_netcdf(me_cache_file)
        
        me_loaded = True
        print("Reanalysis data processed and cached successfully.")
    except Exception as e:
        print(f"Warning: Could not load Reanalysis IC data. Error: {e}")
        me_loaded = False

# --- Process IMERG ---
print("\nLoading IMERG Observation Data...")
if os.path.exists(imerge_cache_file):
    print(f"Loading IMERG mean from cache: {imerge_cache_file}")
    imerge_mean = xr.open_dataarray(imerge_cache_file)
    imerge_loaded = True
else:
    try:
        print(f"Opening IMERG files from: {imerg_path} (Parallel=False, Engine=netcdf4) ...")
        all_imerg_files = sorted(glob.glob(imerg_path))
        start_yyyymmdd = start_date.replace('-', '')
        end_yyyymmdd   = end_date.replace('-', '')
        imerg_files = [
            f for f in all_imerg_files
            if any(
                yyyymmdd >= start_yyyymmdd and yyyymmdd <= end_yyyymmdd
                for yyyymmdd in [os.path.basename(f).split('.')[4].split('-')[0]]
            )
        ]
        print(f"Found {len(imerg_files)} IMERG files in date range ({start_date} to {end_date}).")
        imerg_ds = xr.open_mfdataset(imerg_files, engine='netcdf4', combine='by_coords', parallel=False)
        
        var_name = [v for v in imerg_ds.data_vars if 'precip' in v.lower() or 'pr' in v.lower()][0]
        print(f"Identified IMERG precipitation variable as: {var_name}")
        print("Computing IMERG precipitation...")
        imerg_pr = imerg_ds[var_name].compute(scheduler='synchronous')
        
        # mm/hr to mm/day
        print("Scaling IMERG precipitation from mm/hr to mm/day...")
        imerg_pr = imerg_pr * 24.0

        print(f"Slicing IMERG data for time period: {start_date} to {end_date}...")
        imerg_pr = imerg_pr.sel(time=slice(start_date, end_date))
        
        # Downscale IMERG from 0.1 degree to ~1 degree using mass-conserving block average
        print("Performing mass-conserving 10x10 block average of IMERG from 0.1 to ~1-degree...")
        imerg_pr = imerg_pr.coarsen(lat=10, lon=10, boundary='trim').mean()
        print("IMERG spatial coarsening complete.")

        # Time mean
        imerg_mean = imerg_pr.mean(dim='time')

        print(f"Saving IMERG mean to cache: {imerge_cache_file}")
        imerge_mean.to_netcdf(imerge_cache_file)

        imerge_loaded = True
        print("IMERG data processed and cached successfully.")
    except Exception as e:
        print(f"Warning: Could not load or process IMERG data. Error: {e}")
        imerge_loaded = False
        imerg_mean = None

# ==========================================
# Plotting
# ==========================================
if me_loaded and imerge_loaded:
    print("\nAligning grids for difference...")
    # Use interp to get the arrays on the exact same lats and lons smoothly
    imerg_mean_aligned = imerg_mean.interp(lat=me_mean.lat, lon=me_mean.lon, method='nearest')
    diff = me_mean - imerg_mean_aligned

    print("\nGenerating Robinson Map Plot...")
    fig = plt.figure(figsize=(12, 14))
    
    proj = ccrs.Robinson(central_longitude=180)
    
    # Add cyclic points to avoid white line artifact near 180 longitude
    me_mean_cyc, lon_cyc = add_cyclic_point(me_mean, coord=me_mean.lon)
    imerg_mean_cyc, _ = add_cyclic_point(imerg_mean, coord=imerg_mean.lon)
    diff_cyc, _ = add_cyclic_point(diff, coord=diff.lon)

    # 1. Reanalysis IC Mean
    ax1 = fig.add_subplot(3, 1, 1, projection=proj)
    ax1.coastlines(alpha=0.6)
    ax1.set_global()
    p1 = ax1.contourf(lon_cyc, me_mean.lat, me_mean_cyc, transform=ccrs.PlateCarree(),
                      levels=np.linspace(0, 30, 16), cmap='Blues', extend='max')
    ax1.set_title('(a) Reanalysis IC Mean Precipitation (May 6-31, 2005)', fontsize=14, fontweight='bold')
    plt.colorbar(p1, ax=ax1, orientation='vertical', pad=0.02, shrink=0.8, label='mm/day')

    # 2. IMERG Obs Mean
    ax2 = fig.add_subplot(3, 1, 2, projection=proj)
    ax2.coastlines(alpha=0.6)
    ax2.set_global()
    p2 = ax2.contourf(lon_cyc, imerg_mean.lat, imerg_mean_cyc, transform=ccrs.PlateCarree(),
                      levels=np.linspace(0, 30, 16), cmap='Blues', extend='max')
    ax2.set_title('(b) IMERG Obs Mean Precipitation (May 6-31, 2005)', fontsize=14, fontweight='bold')
    plt.colorbar(p2, ax=ax2, orientation='vertical', pad=0.02, shrink=0.8, label='mm/day')

    # 3. Difference (Reanalysis - IMERG)
    ax3 = fig.add_subplot(3, 1, 3, projection=proj)
    ax3.coastlines(alpha=0.6)
    ax3.set_global()
    p3 = ax3.contourf(lon_cyc, me_mean.lat, diff_cyc, transform=ccrs.PlateCarree(),
                      levels=np.linspace(-15, 15, 16), cmap='RdBu_r', extend='both')
    ax3.set_title('(c) Difference: Reanalysis minus IMERG', fontsize=14, fontweight='bold')
    plt.colorbar(p3, ax=ax3, orientation='vertical', pad=0.02, shrink=0.8, label='mm/day')

    plt.tight_layout()
    output_filename = 'pr_closeness.png'
    print(f"Saving plot to {output_filename}...")
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved successfully as {output_filename}")
else:
    print("\nCannot generate plot without both Reanalysis and IMERG data.")
