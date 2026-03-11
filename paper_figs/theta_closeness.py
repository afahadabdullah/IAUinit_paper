"""
theta_closeness.py

Recreates the closeness plot from ECCO_bias.ipynb using Robinson map projection.
Caches the raw LLC-gridded data for each exp/init_date to avoid recomputing.

KEY DIFFERENCE from notebook:
- Notebook used positional numpy subtraction: ME426 - theta.sel(time=...).data
  (both arrays must have same number of timesteps; time coords are irrelevant)
- This script does the same via numpy .values
"""

import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import matplotlib.colors as mcolors
import ecco_v4_py as ecco

import warnings
warnings.filterwarnings("ignore")

# ==========================================
# Paths and Configuration
# ==========================================
EXP_LOC     = '/nobackupp27/afahad/exp/IAU_exp/'   # all exps are in IAU_exp
INPUT_DIR   = '/nobackupp27/afahad/GEOSMITgcmFiles/mit_input_llc90_02'
ECCO_THETA_DIR = '/nobackupp27/afahad/exp/script_replay_AGU/data/ecco/'

# Cache directory
cache_dir = 'data'
os.makedirs(cache_dir, exist_ok=True)

# Only 0506 has sufficient binary output files (697 files vs 12 for 0426, 24 for 0511)
init_dates = ['0506']

# ECCO THETA time slice to subtract for each init date
# 0506: model starts May 6, so THETA should be from May 6 onward
# Model runs are at 12H freq, so 120 files = 60 days
theta_slices = {
    '0506': ('2005-05-06', '2005-07-05'),
}

# ==========================================
# Helper: LLC grid to regular lat/lon grid
# ==========================================
def llc2grd(theta_arr, nz=1):
    """Regrid LLC array (nt, nz, ntile, nj, ni) to regular (nt, nlat, nlon)."""
    ecco_grid = ecco.load_ecco_grid_nc(INPUT_DIR, 'ECCO-GRID.nc')

    new_lat = np.arange(-90, 91, 1); nlat = len(new_lat)
    new_lon = np.arange(-180, 180, 1); nlon = len(new_lon)

    nt = len(theta_arr)
    out = np.zeros((nt, nz, nlat, nlon))

    for i in range(nt):
        for j in range(nz):
            _, _, _, _, out[i, j] = ecco.resample_to_latlon(
                ecco_grid.XC, ecco_grid.YC, theta_arr[i, j],
                -90, 91, 1, -180, 180, 1,
                fill_value=np.NaN,
                mapping_method='nearest_neighbor',
                radius_of_influence=120000)

    return np.squeeze(out), new_lon, new_lat


# ==========================================
# Data Loading with Caching
# ==========================================

def load_ecco_theta():
    """Load ECCO THETA and regrid to lat/lon. Returns xr.DataArray (time, lat, lon)."""
    cache = os.path.join(cache_dir, 'ecco_theta_gridded.nc')
    if os.path.exists(cache):
        print("Loading THETA from cache...")
        return xr.open_dataarray(cache)

    print("Computing THETA from ECCO files...")
    ds_ecco = xr.open_mfdataset(ECCO_THETA_DIR + 'THETA_2005_0*.nc')
    eTHETA = ds_ecco.THETA[:, 0:1].compute().values   # (nt, 1, ntile, nj, ni)

    val, lon, lat = llc2grd(eTHETA, nz=1)             # (nt, nlat, nlon)
    theta = xr.DataArray(
        val,
        coords={"time": ds_ecco.time.values, "lat": lat, "lon": lon},
        dims=["time", "lat", "lon"]
    )
    theta.to_netcdf(cache)
    print(f"THETA cached to {cache}")
    return theta


def readmit_raw(exp_name, start_date='20050501', nfiles=120, freq='12H',
                nz=50, nf=6, ni=90, nj=90, ntile=13, var=1):
    """
    Read raw LLC binary files for one experiment, return (nt, nz, ntile, nj, ni) array.
    Mirrors the original readmit() from the notebook but uses absolute paths.
    """
    expdir = os.path.join(EXP_LOC, exp_name, 'mit_output')
    if not os.path.exists(expdir):
        print(f"  ERROR: {expdir} does not exist")
        return None, None

    pattern = os.path.join(expdir, 'state_3d_set1*.data')
    files = np.array(sorted(glob.glob(pattern))[:nfiles])
    if len(files) == 0:
        print(f"  WARNING: No files in {expdir}")
        return None, None

    time = pd.date_range(start_date, periods=len(files), freq=freq)
    nt = len(files)
    data_out = np.zeros((nt, nz, ntile, nj, ni), dtype=np.float32)
    data_out[:] = np.nan

    print(f"  Reading {nt} files for {exp_name}...")
    for i, fpath in enumerate(files):
        dname, fname = os.path.dirname(fpath), os.path.basename(fpath)
        try:
            raw = ecco.read_llc_to_tiles(dname, fname, nk=-1, nl=-1)
            raw = np.reshape(raw, (nf, nz, ntile, nj, ni))
            data_out[i] = raw[var]
        except Exception as e:
            print(f"  Failed {fname}: {e}")
            return None, None

    return data_out, time


def load_model_daily(exp_prefix, idate):
    """
    Load daily-mean surface theta for one exp/init_date.
    Returns xr.DataArray (time_daily, lat, lon).
    """
    cache = os.path.join(cache_dir, f'{exp_prefix}_{idate}_daily.nc')
    if os.path.exists(cache):
        print(f"  Loading {exp_prefix}_{idate} from cache...")
        return xr.open_dataarray(cache)

    print(f"  Computing {exp_prefix}_{idate}...")
    raw, time = readmit_raw(f'{exp_prefix}{idate}')
    if raw is None:
        return None

    # Take surface level only
    raw = raw[:, 0:1]                           # (nt, 1, ntile, nj, ni)
    val, lon, lat = llc2grd(raw, nz=1)          # (nt, nlat, nlon)

    da = xr.DataArray(val, coords={"time": time, "lat": lat, "lon": lon},
                      dims=["time", "lat", "lon"])
    da = da.resample(time='1D').mean()          # daily means
    da.to_netcdf(cache)
    print(f"  Cached to {cache}")
    return da


# ==========================================
# Main: Load data and compute biases
# ==========================================
print("Loading ECCO THETA...")
theta = load_ecco_theta()
print(f"THETA shape: {theta.shape},  time: {theta.time.values[[0,-1]]}")

BME_list = []
BRP_list = []

for idate in init_dates:
    print(f"\n=== Init date {idate} ===")
    me = load_model_daily('GEOSMIT_ME', idate)
    rp = load_model_daily('GEOSMIT_RP', idate)
    if me is None or rp is None:
        print(f"  Skipping {idate}")
        continue

    t0, t1 = theta_slices[idate]
    theta_np = theta.sel(time=slice(t0, t1)).values    # numpy (nt_th, nlat, nlon)
    me_np    = me.values                               # numpy (nt_me, nlat, nlon)
    rp_np    = rp.values

    t_len = min(len(me_np), len(theta_np))
    print(f"  me: {me_np.shape}, theta: {theta_np.shape}, using t_len={t_len}")

    # NOTEBOOK-STYLE: positional numpy subtraction
    bme = me_np[:t_len] - theta_np[:t_len]             # (t_len, nlat, nlon)
    brp = rp_np[:t_len] - theta_np[:t_len]

    # Wrap back in xr.DataArray re-using me's time coords for time slicing later
    me_time = me.time.values[:t_len]
    bme_da = xr.DataArray(bme, coords={"time": me_time, "lat": me.lat, "lon": me.lon},
                          dims=["time", "lat", "lon"])
    brp_da = xr.DataArray(brp, coords={"time": me_time, "lat": rp.lat, "lon": rp.lon},
                          dims=["time", "lat", "lon"])

    BME_list.append(bme_da)
    BRP_list.append(brp_da)

if len(BME_list) == 0:
    print("No valid data. Exiting.")
    exit(1)

# Single experiment — just use as is (no averaging needed)
BME = BME_list[0]
BRP = BRP_list[0]

# Apply same first-timestep zeroing as notebook: BME[0,:,:] = 0
BME[0] = 0.0

print(f"\nBME shape: {BME.shape}, time: {BME.time.values[[0,-1]]}")
print(f"BRP shape: {BRP.shape}")

# ==========================================
# Closeness per week  |BME| - |BRP|
# ==========================================
def closeness(t0, t1):
    a = np.abs(BME.sel(time=slice(t0, t1))).mean(dim='time')
    b = np.abs(BRP.sel(time=slice(t0, t1))).mean(dim='time')
    return a - b

w1  = closeness('2005-05-01', '2005-05-07')
w2  = closeness('2005-05-08', '2005-05-14')
w34 = closeness('2005-05-15', '2005-05-30')
w58 = closeness('2005-06-01', '2005-06-30')

for name, arr in [('w1',w1),('w2',w2),('w34',w34),('w58',w58)]:
    print(f"{name}: min={np.nanmin(arr.values):.3f}  max={np.nanmax(arr.values):.3f}")

# ==========================================
# Plotting – Robinson projection
# ==========================================
print("\nGenerating Robinson projection plots...")

levels = np.arange(-0.9, 0.91, 0.2)
cmap   = 'RdBu_r'   # Blue = ME closer, Red = IAU closer

proj = ccrs.Robinson(central_longitude=180)
fig, axes = plt.subplots(2, 2, figsize=(16, 10), subplot_kw={'projection': proj})

titles = [
    'Week 1 closeness (Blue: ME Closer, Red: IAU closer)',
    'Week 2 closeness (Blue: ME Closer, Red: IAU closer)',
    'Week 3-4 closeness (Blue: ME Closer, Red: IAU closer)',
    'Week 5-8 closeness (Blue: ME Closer, Red: IAU closer)',
]
datasets = [w1, w2, w34, w58]

plots = []
for ax, da, title in zip(axes.flat, datasets, titles):
    ax.set_global()
    ax.coastlines(linewidth=0.5, alpha=0.6)
    ax.set_title(title, fontsize=11, fontweight='bold')
    da_cyc, lon_cyc = add_cyclic_point(da, coord=da.lon)
    p = ax.contourf(lon_cyc, da.lat, da_cyc,
                    transform=ccrs.PlateCarree(),
                    levels=levels, cmap=cmap, extend='both')
    plots.append(p)

fig.subplots_adjust(right=0.85, wspace=0.05, hspace=0.15)
cbar_ax = fig.add_axes([0.88, 0.2, 0.02, 0.6])
cbar = fig.colorbar(plots[0], cax=cbar_ax, ticks=levels)
cbar.set_label('°C  (Blue: ME closer to ECCO | Red: IAU closer to ECCO)')

out = 'theta_closeness.png'
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f"Saved: {out}")
