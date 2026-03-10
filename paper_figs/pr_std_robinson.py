import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from matplotlib.colors import TwoSlopeNorm
import matplotlib.colors as mcolors
from scipy import stats

# ==========================================
# File Paths and Cache Configuration
# ==========================================
init_dates = ['0426', '0506', '0511']
imerg_path = '/nobackupp27/afahad/project/IAUinit_paper/data/3B-MO.MS.MRG*200505*.nc4'

# Cache Directory
cache_dir = 'data'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
    print(f"Created cache directory: {cache_dir}")

imerge_cache_file  = os.path.join(cache_dir, 'imerge_mean_200505.nc4')

# ==========================================
# Data Loading and Processing
# ==========================================

start_date = '2005-05-06'
end_date = '2005-05-31'

def load_and_process(paths, cache_file_mean, cache_file_std):
    if os.path.exists(cache_file_mean) and os.path.exists(cache_file_std):
        print(f"  Loading from cache: {cache_file_mean}, {cache_file_std}")
        return xr.open_dataarray(cache_file_mean), xr.open_dataarray(cache_file_std)
        
    print(f"  Opening files from: {paths[0]}...")
    files = sorted([f for p in paths for f in glob.glob(p)])
    print(f"  Found {len(files)} files.")
    if len(files) == 0:
        print("  Warning: No files found.")
        return None, None
        
    ds = xr.open_mfdataset(files, engine='netcdf4', combine='by_coords', parallel=False)
    print("  Computing precipitation...")
    pr = ds.PRECTOT.compute(scheduler='synchronous')
    
    # Convert from kg/m^2/s to mm/day
    pr = pr * 86400 

    # Slice for the analysis period
    pr = pr.sel(time=slice(start_date, end_date))
    
    # Compute time mean and standard deviation
    pr_mean = pr.mean(dim='time')
    pr_std = pr.std(dim='time')
    
    print(f"  Saving to cache: {cache_file_mean}, {cache_file_std}")
    pr_mean.to_netcdf(cache_file_mean)
    pr_std.to_netcdf(cache_file_std)
    
    return pr_mean, pr_std

def get_ensemble(prefix):
    mean_list = []
    std_list = []
    print(f"\nProcessing Ensemble Data ({prefix})...")
    for idate in init_dates:
        print(f" \n--- Init Date: {idate} ---")
        paths = [
            f'/nobackupp27/afahad/exp/IAU_exp/{prefix}{idate}/holding/geosgcm_surf/200505/*surf*200505*z.nc4',
            f'/nobackupp27/afahad/exp/IAU_exp/{prefix}{idate}/holding/geosgcm_surf/200506/*surf*200506*z.nc4'
        ]
        c_mean = os.path.join(cache_dir, f'{prefix.lower()}_mean_200505_{idate}.nc4')
        c_std = os.path.join(cache_dir, f'{prefix.lower()}_std_200505_{idate}.nc4')
        
        m, s = load_and_process(paths, c_mean, c_std)
        if m is not None and s is not None:
            mean_list.append(m)
            std_list.append(s)
            
    if not mean_list:
        return None, None, None
        
    print(f"\nComputing ensemble mean across {len(mean_list)} members for {prefix}...")
    # Concatenate along a new dimension and compute ensemble mean
    ens_members = xr.concat(mean_list, dim='ens')
    ens_mean = ens_members.mean(dim='ens')
    ens_std = xr.concat(std_list, dim='ens').mean(dim='ens')
    
    return ens_mean, ens_std, ens_members

me_mean, me_std, me_members = get_ensemble('GEOSMIT_ME')
rp_mean, rp_std, rp_members = get_ensemble('GEOSMIT_RP')

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
    
    # --- Significance Testing (95%) ---
    print("\nComputing significance masks (p < 0.05)...")
    # (b) Rean IC vs IAU IC
    _, p_val_b = stats.ttest_ind(me_members.values, rp_members.values, axis=0, nan_policy='omit')
    diff_mean = diff_mean.where(p_val_b < 0.05)

    # Align IMERG grids for diff calculation
    print("Aligning IMERG grids for closeness...")
    imerg_mean_aligned = imerg_mean.interp(lat=me_mean.lat, lon=me_mean.lon, method='nearest')
    
    # Fill any NaNs created at the wrapping boundary during interpolation
    imerg_mean_aligned = imerg_mean_aligned.bfill(dim='lon').ffill(dim='lon')
    
    # (c) Rean IC vs IMERG
    diff_mean_imerg = me_mean - imerg_mean_aligned
    _, p_val_c = stats.ttest_1samp(me_members.values, imerg_mean_aligned.values, axis=0, nan_policy='omit')
    diff_mean_imerg = diff_mean_imerg.where(p_val_c < 0.05)
    
    # (d) Closeness ( |Rean - Obs| - |IAU - Obs| )
    # Compute closeness for each member to allow for significance testing
    close_members = np.abs(me_members - imerg_mean_aligned) - np.abs(rp_members - imerg_mean_aligned)
    closeness = close_members.mean(dim='ens')
    _, p_val_d = stats.ttest_1samp(close_members.values, 0, axis=0, nan_policy='omit')
    closeness = closeness.where(p_val_d < 0.05)
    print("Significance masking complete.")

    # ==========================================
    # Plotting
    # ==========================================
    print("\nGenerating Robinson projection plot...")

    proj = ccrs.Robinson(central_longitude=180)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), subplot_kw={'projection': proj})

    def format_axis(ax, title):
        ax.set_global()
        ax.coastlines(linewidth=0.5, alpha=0.6)
        ax.set_title(title, fontsize=14, fontweight='bold')

    ax_me_mean = axes[0, 0]
    ax_diff_mean = axes[0, 1]
    ax_diff_imerg = axes[1, 0]
    ax_closeness = axes[1, 1]

    format_axis(ax_me_mean, '(a) Reanalysis IC Ens Mean Precip')
    format_axis(ax_diff_mean, '(b) Ens Mean Diff (Reanalysis IC - IAU IC)')
    format_axis(ax_diff_imerg, '(c) Ens Mean Diff (Reanalysis IC - IMERG)')
    format_axis(ax_closeness, '(d) Ens Closeness: |Rean - Obs| - |IAU - Obs|')

    # Add cyclic points 
    me_mean_cyc, lon_cyc = add_cyclic_point(me_mean, coord=me_mean.lon)
    diff_mean_cyc, _ = add_cyclic_point(diff_mean, coord=diff_mean.lon)
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

    # Plot (c): Diff IMERG Mean
    p5 = ax_diff_imerg.contourf(lon_cyc, diff_mean_imerg.lat, diff_mean_imerg_cyc, transform=ccrs.PlateCarree(),
                               levels=diff_mean_levels, cmap='RdBu_r', extend='both', norm=norm_mean)
    fig.colorbar(p5, ax=ax_diff_imerg, orientation='horizontal', shrink=0.8, pad=0.05, label='mm/day', ticks=diff_mean_levels)

    # Plot (d): Closeness
    # Custom colormap: PiYG, but white between -1 and 1
    close_levels = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
    cmap_base = plt.get_cmap('PiYG')
    custom_colors = np.vstack((cmap_base(np.linspace(0, 0.4, 4)), 
                               np.array([[1, 1, 1, 1]]), 
                               cmap_base(np.linspace(0.6, 1, 4))))
    cmap_close = mcolors.ListedColormap(custom_colors)
    cmap_close.set_under(cmap_base(0.0))
    cmap_close.set_over(cmap_base(1.0))
    norm_close = mcolors.BoundaryNorm(close_levels, cmap_close.N)

    p6 = ax_closeness.contourf(lon_cyc, closeness.lat, closeness_cyc, transform=ccrs.PlateCarree(),
                              levels=close_levels, cmap=cmap_close, extend='both', norm=norm_close)
    cbar6 = fig.colorbar(p6, ax=ax_closeness, orientation='horizontal', shrink=0.8, pad=0.05, ticks=close_levels)
    cbar6.set_label('mm/day (Pink: Reanalysis IC closer, Green: IAU IC closer)')

    plt.tight_layout()

    output_filename = 'pr_std_robinson.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {output_filename}")
else:
    print("Cannot generate plot due to missing data.")
