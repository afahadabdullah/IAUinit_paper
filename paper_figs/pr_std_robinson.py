import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import TwoSlopeNorm

# ==========================================
# File Paths
# ==========================================
# Reanalysis Initialized (ME506) and Replay (RP506)
me_paths = [
    '/nobackupp27/afahad/exp/IAU_exp/GEOSMIT_ME0506/holding/geosgcm_surf/200505/*surf*200505*z.nc4',
    '/nobackupp27/afahad/exp/IAU_exp/GEOSMIT_ME0506/holding/geosgcm_surf/200506/*surf*200506*z.nc4'
]

rp_paths = [
    '/nobackupp27/afahad/exp/GEOSMIT_RP0506/holding/geosgcm_surf/200505/*surf*200505*z.nc4',
    '/nobackupp27/afahad/exp/GEOSMIT_RP0506/holding/geosgcm_surf/200506/*surf*200506*z.nc4'
]

# ==========================================
# Data Loading and Processing
# ==========================================

start_date = '2005-05-06'
end_date = '2005-05-31'

def load_and_process(paths):
    print(f"Opening files from: {paths} ...")
    files = sorted([f for p in paths for f in glob.glob(p)])
    print(f"Found {len(files)} files.")
    if len(files) == 0:
        print("Warning: No files found. Make sure paths are correct on PFE.")
        return None
        
    ds = xr.open_mfdataset(files, engine='netcdf4', combine='by_coords', parallel=False)
    print("Computing precipitation...")
    pr = ds.PRECTOT.compute(scheduler='synchronous')
    
    # Convert from kg/m^2/s to mm/day
    print("Converting units to mm/day...")
    pr = pr * 86400 

    # Slice for the analysis period
    print(f"Slicing data for time period: {start_date} to {end_date}...")
    pr = pr.sel(time=slice(start_date, end_date))
    
    # Compute temporal standard deviation
    pr_std = pr.std(dim='time')
    return pr_std

print("Processing Reanalysis IC Data (ME506P)...")
me_std = load_and_process(me_paths)

print("\nProcessing Replay Data (RP506P)...")
rp_std = load_and_process(rp_paths)

if me_std is not None and rp_std is not None:
    # Compute difference: Replay - Reanalysis IC
    diff = rp_std - me_std

    # ==========================================
    # Plotting
    # ==========================================
    print("\nGenerating Robinson projection plot...")

    fig, axes = plt.subplots(3, 1, figsize=(10, 15), subplot_kw={'projection': ccrs.Robinson(central_longitude=0)})

    # Function to format axes
    def format_axis(ax, title):
        ax.set_global()
        ax.coastlines(linewidth=0.5, color='gray')
        ax.set_title(title, fontsize=14)

    format_axis(axes[0], 'Temporal Std Precip: Reanalysis IC (ME506)')
    format_axis(axes[1], 'Temporal Std Precip: Replay (RP506)')
    format_axis(axes[2], 'Difference: Replay - Reanalysis IC')

    # Plot 1: ME Std
    im1 = axes[0].pcolormesh(me_std.lon, me_std.lat, me_std, transform=ccrs.PlateCarree(),
                             cmap='YlGnBu', vmin=0, vmax=25)
    cbar1 = fig.colorbar(im1, ax=axes[0], orientation='horizontal', shrink=0.7, pad=0.05)
    cbar1.set_label('Precipitation Std (mm/day)')

    # Plot 2: RP Std
    im2 = axes[1].pcolormesh(rp_std.lon, rp_std.lat, rp_std, transform=ccrs.PlateCarree(),
                             cmap='YlGnBu', vmin=0, vmax=25)
    cbar2 = fig.colorbar(im2, ax=axes[1], orientation='horizontal', shrink=0.7, pad=0.05)
    cbar2.set_label('Precipitation Std (mm/day)')

    # Plot 3: Difference
    # Using RdBu_r so that negative values (reduced variance) are blue, positive are red.
    norm = TwoSlopeNorm(vmin=-5, vcenter=0, vmax=5)
    im3 = axes[2].pcolormesh(diff.lon, diff.lat, diff, transform=ccrs.PlateCarree(),
                             cmap='RdBu_r', norm=norm)
    cbar3 = fig.colorbar(im3, ax=axes[2], orientation='horizontal', shrink=0.7, pad=0.05)
    cbar3.set_label('Difference (mm/day)')

    plt.tight_layout()

    output_filename = 'pr_std_robinson.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {output_filename}")
else:
    print("Cannot generate plot due to missing data.")
