import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import TwoSlopeNorm

# ==========================================
# File Paths and Cache Configuration
# ==========================================
# Reanalysis Initialized (ME506) and Replay (RP506)
me_paths = [
    '/nobackupp27/afahad/exp/IAU_exp/GEOSMIT_ME0506/holding/geosgcm_surf/200505/*surf*200505*z.nc4',
    '/nobackupp27/afahad/exp/IAU_exp/GEOSMIT_ME0506/holding/geosgcm_surf/200506/*surf*200506*z.nc4'
]

rp_paths = [
    '/nobackupp27/afahad/exp/IAU_exp/GEOSMIT_RP0506/holding/geosgcm_surf/200505/*surf*200505*z.nc4',
    '/nobackupp27/afahad/exp/IAU_exp/GEOSMIT_RP0506/holding/geosgcm_surf/200506/*surf*200506*z.nc4'
]

# Cache Directory
cache_dir = 'data'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
    print(f"Created cache directory: {cache_dir}")

me_cache_file_mean = os.path.join(cache_dir, 'me_mean_200505.nc4')
rp_cache_file_mean = os.path.join(cache_dir, 'rp_mean_200505.nc4')
me_cache_file_std = os.path.join(cache_dir, 'me_std_200505.nc4')
rp_cache_file_std = os.path.join(cache_dir, 'rp_std_200505.nc4')

# ==========================================
# Data Loading and Processing
# ==========================================

start_date = '2005-05-06'
end_date = '2005-05-31'

def load_and_process(paths, cache_file_mean, cache_file_std):
    if os.path.exists(cache_file_mean) and os.path.exists(cache_file_std):
        print(f"Loading mean and std from cache: {cache_file_mean}, {cache_file_std}")
        return xr.open_dataarray(cache_file_mean), xr.open_dataarray(cache_file_std)
        
    print(f"Opening files from: {paths} ...")
    files = sorted([f for p in paths for f in glob.glob(p)])
    print(f"Found {len(files)} files.")
    if len(files) == 0:
        print("Warning: No files found. Make sure paths are correct on PFE.")
        return None, None
        
    ds = xr.open_mfdataset(files, engine='netcdf4', combine='by_coords', parallel=False)
    print("Computing precipitation...")
    pr = ds.PRECTOT.compute(scheduler='synchronous')
    
    # Convert from kg/m^2/s to mm/day
    print("Converting units to mm/day...")
    pr = pr * 86400 

    # Slice for the analysis period
    print(f"Slicing data for time period: {start_date} to {end_date}...")
    pr = pr.sel(time=slice(start_date, end_date))
    
    # Compute time mean and standard deviation
    pr_mean = pr.mean(dim='time')
    pr_std = pr.std(dim='time')
    
    print(f"Saving mean and std calculation to cache: {cache_file_mean}, {cache_file_std}")
    pr_mean.to_netcdf(cache_file_mean)
    pr_std.to_netcdf(cache_file_std)
    
    return pr_mean, pr_std

print("Processing Reanalysis IC Data (ME506P)...")
me_mean, me_std = load_and_process(me_paths, me_cache_file_mean, me_cache_file_std)

print("\nProcessing Replay Data (RP506P)...")
rp_mean, rp_std = load_and_process(rp_paths, rp_cache_file_mean, rp_cache_file_std)

if me_mean is not None and me_std is not None and rp_mean is not None and rp_std is not None:
    # Compute difference: IAU (Reanalysis IC) - Replay
    diff_mean = me_mean - rp_mean
    diff_std = me_std - rp_std

    # ==========================================
    # Plotting
    # ==========================================
    print("\nGenerating Robinson projection plot...")

    proj = ccrs.Robinson(central_longitude=180)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), subplot_kw={'projection': proj})

    # Function to format axes
    def format_axis(ax, title):
        ax.set_global()
        ax.coastlines(linewidth=0.5, alpha=0.6)
        ax.set_title(title, fontsize=14, fontweight='bold')

    ax_me_mean = axes[0, 0]
    ax_diff_mean = axes[0, 1]
    ax_me_std = axes[1, 0]
    ax_diff_std = axes[1, 1]

    format_axis(ax_me_mean, '(a) Reanalysis IC Mean Precip')
    format_axis(ax_diff_mean, '(b) Mean Difference (IAU - Replay)')
    format_axis(ax_me_std, '(c) Reanalysis IC Precip Std')
    format_axis(ax_diff_std, '(d) Std Difference (IAU - Replay)')

    # Plot 1: ME Mean
    p1 = ax_me_mean.contourf(me_mean.lon, me_mean.lat, me_mean, transform=ccrs.PlateCarree(),
                             levels=np.linspace(0, 30, 16), cmap='Blues', extend='max')
    fig.colorbar(p1, ax=ax_me_mean, orientation='horizontal', shrink=0.8, pad=0.05, label='mm/day')

    # Plot 2: Diff Mean
    norm_mean = TwoSlopeNorm(vmin=-15, vcenter=0, vmax=15)
    p2 = ax_diff_mean.contourf(diff_mean.lon, diff_mean.lat, diff_mean, transform=ccrs.PlateCarree(),
                               levels=np.linspace(-15, 15, 16), cmap='RdBu_r', extend='both', norm=norm_mean)
    fig.colorbar(p2, ax=ax_diff_mean, orientation='horizontal', shrink=0.8, pad=0.05, label='mm/day')

    # Plot 3: ME Std
    p3 = ax_me_std.contourf(me_std.lon, me_std.lat, me_std, transform=ccrs.PlateCarree(),
                            levels=np.linspace(0, 25, 16), cmap='YlGnBu', extend='max')
    fig.colorbar(p3, ax=ax_me_std, orientation='horizontal', shrink=0.8, pad=0.05, label='mm/day')

    # Plot 4: Diff Std
    norm_std = TwoSlopeNorm(vmin=-5, vcenter=0, vmax=5)
    p4 = ax_diff_std.contourf(diff_std.lon, diff_std.lat, diff_std, transform=ccrs.PlateCarree(),
                              levels=np.linspace(-5, 5, 21), cmap='RdBu_r', extend='both', norm=norm_std)
    fig.colorbar(p4, ax=ax_diff_std, orientation='horizontal', shrink=0.8, pad=0.05, label='mm/day')

    plt.tight_layout()

    output_filename = 'pr_std_robinson.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {output_filename}")
else:
    print("Cannot generate plot due to missing data.")
