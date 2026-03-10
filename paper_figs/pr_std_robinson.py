import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from matplotlib.colors import TwoSlopeNorm

# ==========================================
# File Paths and Cache Configuration
# ==========================================
# Reanalysis Initialized (ME506) and IAU IC (RP506)
me_paths = [
    '/nobackupp27/afahad/exp/IAU_exp/GEOSMIT_ME0506/holding/geosgcm_surf/200505/*surf*200505*z.nc4',
    '/nobackupp27/afahad/exp/IAU_exp/GEOSMIT_ME0506/holding/geosgcm_surf/200506/*surf*200506*z.nc4'
]

rp_paths = [
    '/nobackupp27/afahad/exp/IAU_exp/GEOSMIT_RP0506/holding/geosgcm_surf/200505/*surf*200505*z.nc4',
    '/nobackupp27/afahad/exp/IAU_exp/GEOSMIT_RP0506/holding/geosgcm_surf/200506/*surf*200506*z.nc4'
]

imerg_path = '/nobackupp27/afahad/project/IAUinit_paper/data/3B-MO.MS.MRG*200505*.nc4'

# Cache Directory
cache_dir = 'data'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
    print(f"Created cache directory: {cache_dir}")

me_cache_file_mean = os.path.join(cache_dir, 'me_mean_200505.nc4')
rp_cache_file_mean = os.path.join(cache_dir, 'rp_mean_200505.nc4')
me_cache_file_std = os.path.join(cache_dir, 'me_std_200505.nc4')
rp_cache_file_std = os.path.join(cache_dir, 'rp_std_200505.nc4')
imerge_cache_file  = os.path.join(cache_dir, 'imerge_mean_200505.nc4')

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

print("\nProcessing IAU IC Data (RP506P)...")
rp_mean, rp_std = load_and_process(rp_paths, rp_cache_file_mean, rp_cache_file_std)

# --- Process IMERG Mean ---
print("\nLoading IMERG Observation Data...")
if os.path.exists(imerge_cache_file):
    print(f"Loading IMERG mean from cache: {imerge_cache_file}")
    imerg_mean = xr.open_dataarray(imerge_cache_file)
else:
    try:
        print(f"Opening IMERG monthly files from: {imerg_path} ...")
        imerg_files = sorted(glob.glob(imerg_path))
        print(f"Found {len(imerg_files)} IMERG files.")
        
        imerg_ds = xr.open_mfdataset(imerg_files, engine='netcdf4', combine='by_coords', parallel=False)
        var_name = [v for v in imerg_ds.data_vars if 'precip' in v.lower() or 'pr' in v.lower()][0]
        imerg_pr = imerg_ds[var_name].compute(scheduler='synchronous')
        imerg_pr = imerg_pr * 24.0 # mm/hr to mm/day
        
        # Monthly mean data already represents the month, just do spatial mass-conserving coarsen
        # The user's new file has higher resolution (0.1 deg) so coarsen 10x10 is appropriate to get ~1deg
        imerg_pr = imerg_pr.coarsen(lat=10, lon=10, boundary='trim').mean()
        imerg_mean = imerg_pr.mean(dim='time') if 'time' in imerg_pr.dims else imerg_pr
        imerg_mean.to_netcdf(imerge_cache_file)
        print("IMERG data processed and cached successfully.")
    except Exception as e:
        print(f"Warning: Could not load or process IMERG data. Error: {e}")
        imerg_mean = None


if me_mean is not None and me_std is not None and rp_mean is not None and rp_std is not None and imerg_mean is not None:
    # Compute difference: Reanalysis IC - IAU IC
    diff_mean = me_mean - rp_mean
    diff_std = me_std - rp_std
    
    # Align IMERG grids for diff calculation
    print("\nAligning IMERG grids for closeness...")
    imerg_mean_aligned = imerg_mean.interp(lat=me_mean.lat, lon=me_mean.lon, method='nearest')
    
    # Compute Panel (e): Mean diff (Reanalysis IC - IMERG)
    diff_mean_imerg = me_mean - imerg_mean_aligned
    
    # Compute Panel (f): Closeness ( |Reanalysis IC - IMERG| - |IAU IC - IMERG| )
    closeness = np.abs(me_mean - imerg_mean_aligned) - np.abs(rp_mean - imerg_mean_aligned)

    # ==========================================
    # Plotting
    # ==========================================
    print("\nGenerating Robinson projection plot...")

    proj = ccrs.Robinson(central_longitude=180)
    fig, axes = plt.subplots(3, 2, figsize=(16, 15), subplot_kw={'projection': proj})

    def format_axis(ax, title):
        ax.set_global()
        ax.coastlines(linewidth=0.5, alpha=0.6)
        ax.set_title(title, fontsize=14, fontweight='bold')

    ax_me_mean = axes[0, 0]
    ax_diff_mean = axes[0, 1]
    ax_me_std = axes[1, 0]
    ax_diff_std = axes[1, 1]
    ax_diff_imerg = axes[2, 0]
    ax_closeness = axes[2, 1]

    format_axis(ax_me_mean, '(a) Reanalysis IC Mean Precip')
    format_axis(ax_diff_mean, '(b) Mean Difference (Reanalysis IC - IAU IC)')
    format_axis(ax_me_std, '(c) Reanalysis IC Precip Std')
    format_axis(ax_diff_std, '(d) Std Difference (Reanalysis IC - IAU IC)')
    format_axis(ax_diff_imerg, '(e) Mean Difference (Reanalysis IC - IMERG)')
    format_axis(ax_closeness, '(f) Closeness: |Rean - IMERG| - |IAU IC - IMERG|')

    # Add cyclic points 
    me_mean_cyc, lon_cyc = add_cyclic_point(me_mean, coord=me_mean.lon)
    diff_mean_cyc, _ = add_cyclic_point(diff_mean, coord=diff_mean.lon)
    me_std_cyc, _ = add_cyclic_point(me_std, coord=me_std.lon)
    diff_std_cyc, _ = add_cyclic_point(diff_std, coord=diff_std.lon)
    diff_mean_imerg_cyc, _ = add_cyclic_point(diff_mean_imerg, coord=diff_mean_imerg.lon)
    closeness_cyc, _ = add_cyclic_point(closeness, coord=closeness.lon)

    # Plot (a): ME Mean
    p1 = ax_me_mean.contourf(lon_cyc, me_mean.lat, me_mean_cyc, transform=ccrs.PlateCarree(),
                             levels=np.linspace(0, 30, 16), cmap='Blues', extend='max')
    fig.colorbar(p1, ax=ax_me_mean, orientation='horizontal', shrink=0.8, pad=0.05, label='mm/day')

    # Plot (b): Diff Mean
    diff_mean_levels = np.arange(-13, 15, 2)
    norm_mean = TwoSlopeNorm(vmin=-13, vcenter=0, vmax=14)
    p2 = ax_diff_mean.contourf(lon_cyc, diff_mean.lat, diff_mean_cyc, transform=ccrs.PlateCarree(),
                               levels=diff_mean_levels, cmap='RdBu_r', extend='both', norm=norm_mean)
    fig.colorbar(p2, ax=ax_diff_mean, orientation='horizontal', shrink=0.8, pad=0.05, label='mm/day', ticks=diff_mean_levels)

    # Plot (c): ME Std
    p3 = ax_me_std.contourf(lon_cyc, me_std.lat, me_std_cyc, transform=ccrs.PlateCarree(),
                            levels=np.linspace(0, 25, 16), cmap='YlGnBu', extend='max')
    fig.colorbar(p3, ax=ax_me_std, orientation='horizontal', shrink=0.8, pad=0.05, label='mm/day')

    # Plot (d): Diff Std
    diff_std_levels = np.arange(-7, 8, 2)
    norm_std = TwoSlopeNorm(vmin=-7, vcenter=0, vmax=7)
    p4 = ax_diff_std.contourf(lon_cyc, diff_std.lat, diff_std_cyc, transform=ccrs.PlateCarree(),
                              levels=diff_std_levels, cmap='RdBu_r', extend='both', norm=norm_std)
    fig.colorbar(p4, ax=ax_diff_std, orientation='horizontal', shrink=0.8, pad=0.05, label='mm/day', ticks=diff_std_levels)

    # Plot (e): Diff IMERG Mean
    p5 = ax_diff_imerg.contourf(lon_cyc, diff_mean_imerg.lat, diff_mean_imerg_cyc, transform=ccrs.PlateCarree(),
                               levels=diff_mean_levels, cmap='RdBu_r', extend='both', norm=norm_mean)
    fig.colorbar(p5, ax=ax_diff_imerg, orientation='horizontal', shrink=0.8, pad=0.05, label='mm/day', ticks=diff_mean_levels)

    # Plot (f): Closeness
    close_levels = np.arange(-7, 8, 2)
    norm_close = TwoSlopeNorm(vmin=-7, vcenter=0, vmax=7)
    p6 = ax_closeness.contourf(lon_cyc, closeness.lat, closeness_cyc, transform=ccrs.PlateCarree(),
                              levels=close_levels, cmap='PiYG', extend='both', norm=norm_close)
    cbar6 = fig.colorbar(p6, ax=ax_closeness, orientation='horizontal', shrink=0.8, pad=0.05, ticks=close_levels)
    cbar6.set_label('mm/day (Pink: Reanalysis IC closer, Green: IAU IC closer)')

    plt.tight_layout()

    output_filename = 'pr_std_robinson.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {output_filename}")
else:
    print("Cannot generate plot due to missing data.")
