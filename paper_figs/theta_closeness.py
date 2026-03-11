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
# File Paths and Cache Configuration
# ==========================================
init_dates = ['0426', '0506', '0511']

# Cache Directory
cache_dir = 'data'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
    print(f"Created cache directory: {cache_dir}")

def readmit(exp='GEOSMIT35_ctrl', loc='/nobackupp27/afahad/exp/IAU_exp/', var=1, start_date='20050501', nfiles=120, freq='12H', nz=50, nf=6, ni=90, nj=90, ntile=13, expdir_suffix='../mit_output/'):
    input_dir = '/nobackupp27/afahad/GEOSMITgcmFiles/mit_input_llc90_02'
    
    # Resolve the plot dir and mit_output dir using the exp loc
    # Wait, the original was:
    # os.chdir(loc + exp + '/plot')
    # expdir = '../mit_output/' (relative to the plot dir, so it's loc + exp + '/mit_output/')
    # Let's derive the expdir directly!
    full_expdir = os.path.join(loc, exp, 'mit_output')
    
    if not os.path.exists(full_expdir):
        print(f"Error: Directory {full_expdir} does not exist.")
        return None, None
        
    search_pattern = os.path.join(full_expdir, 'state_3d_set1*.data')
    files = np.array(sorted(glob.glob(search_pattern))[0:nfiles])
    if len(files) == 0:
        print(f"Warning: No files found in {full_expdir}")
        return None, None
        
    time = pd.date_range(start_date, periods=len(files), freq=freq)
    
    djf_files = files
    nt = len(time)
    ndjf = len(djf_files)
    theta_djf = np.zeros((ndjf, nz, ntile, nj, ni))
    theta_djf[:] = np.nan
    
    print(f'reading {ndjf} files for {exp} from {full_expdir}')
    if ndjf == 0:
        print(f"Warning: No files found for {exp} in {full_expdir}")
        return None, None

    for i in range(ndjf):
        # ecco.read_llc_to_tiles expects a directory and a filename base.
        # Since files contain the full path, we extract the dir and short name
        dname = os.path.dirname(djf_files[i])
        fname = os.path.basename(djf_files[i])
        try:
            data = ecco.read_llc_to_tiles(dname, fname, nk=-1, nl=-1)
            data = np.reshape(data, (nf, nz, ntile, nj, ni))
            theta_djf[i, :, :, :] = data[var, :, :, :, :]
        except Exception as e:
            print(f"Failed to read/reshape {fname}: {e}")
            return None, None
        
    return (theta_djf, time)
    
def llc2grd(theta_djf,nz=50):
    input_dir = '/nobackupp27/afahad/GEOSMITgcmFiles/mit_input_llc90_02'
    ecco_grid = ecco.load_ecco_grid_nc(input_dir, 'ECCO-GRID.nc')
    theta_djf_all=np.zeros(theta_djf.shape)
    theta_djf_all[:]=theta_djf

    new_grid_delta_lat = 1
    new_grid_delta_lon = 1
    new_grid_min_lat = -90
    new_grid_max_lat = 91
    new_grid_min_lon = -180
    new_grid_max_lon = 180

    new_lat=np.arange(-90,91,1)
    nlat=len(new_lat)
    new_lon=np.arange(-180,180,1)
    nlon=len(new_lon)

    tt=len(theta_djf)
    theta_djf_alli=np.zeros((tt,nz,nlat,nlon))

    for i in range(tt):
        for j in range(nz):
            new_grid_lon_centers, new_grid_lat_centers,\
            new_grid_lon_edges, new_grid_lat_edges,\
            theta_djf_alli[i,j,:,:] =\
                    ecco.resample_to_latlon(ecco_grid.XC, \
                                            ecco_grid.YC, \
                                            theta_djf_all[i,j,:,:,:],\
                                            new_grid_min_lat, new_grid_max_lat, new_grid_delta_lat,\
                                            new_grid_min_lon, new_grid_max_lon, new_grid_delta_lon,\
                                            fill_value = np.NaN, \
                                            mapping_method = 'nearest_neighbor',
                                            radius_of_influence = 120000)
    return (np.squeeze(theta_djf_alli), new_lon, new_lat)

def get_theta():
    cache_file = os.path.join(cache_dir, 'ecco_theta_gridded.nc')
    if os.path.exists(cache_file):
        print("Loading THETA from cache...")
        return xr.open_dataarray(cache_file)
    print("Computing THETA...")
    dir='/nobackupp27/afahad/exp/script_replay_AGU/data/ecco/'
    eTHETA=xr.open_mfdataset(dir+'THETA_2005_0*.nc').THETA.compute()
    eTHETA=eTHETA[:,0:1,:,:,:].values
    THETA_val,lon,lat=llc2grd(eTHETA,nz=1)
    # The original script loads THETA using time from eTHETA
    eTHETA_xr = xr.open_mfdataset(dir+'THETA_2005_0*.nc')
    theta = xr.DataArray(THETA_val, coords={"time": eTHETA_xr.time.values, "lat": lat, "lon": lon}, dims=["time", "lat", "lon"])
    theta.to_netcdf(cache_file)
    return theta

def get_model_data(exp_prefix, idate):
    cache_file = os.path.join(cache_dir, f'{exp_prefix}_{idate}_theta.nc')
    if os.path.exists(cache_file):
        print(f"Loading {exp_prefix}_{idate} from cache...")
        return xr.open_dataarray(cache_file)
    
    print(f"Computing {exp_prefix}_{idate}...")
    data_val, time = readmit(f'{exp_prefix}{idate}',nz=50,nfiles=120,freq='12H',start_date='20050501')
    if data_val is None:
        return None
        
    data_val = data_val[:,0:1,:,:,:]
    data_val, lon, lat = llc2grd(data_val, nz=1)
    ds_da = xr.DataArray(data_val, coords={"time": time, "lat": lat, "lon": lon}, dims=["time", "lat", "lon"])
    
    resampled_da = ds_da.resample(time='1D').mean()
    resampled_da.to_netcdf(cache_file)
    return resampled_da

print("Starting to load/process data...")
theta = get_theta()

bme_list = []
brp_list = []

idate_to_slice = {
    '0426': slice('2005-04-26', '2005-06-25'),
    '0506': slice('2005-05-06', '2005-07-05'),
    '0511': slice('2005-05-11', '2005-07-10')
}

for idate in init_dates:
    me = get_model_data('GEOSMIT_ME', idate)
    rp = get_model_data('GEOSMIT_RP', idate)
    
    if me is not None and rp is not None:
        theta_slice = theta.sel(time=idate_to_slice[idate])
        
        # truncate/align array dimensions if needed
        t_len = min(len(me), len(theta_slice))
        
        bme = me.copy()
        bme.values[:t_len] = me.values[:t_len] - theta_slice.values[:t_len]
        
        brp = rp.copy()
        brp.values[:t_len] = rp.values[:t_len] - theta_slice.values[:t_len]
        
        bme_list.append(bme)
        brp_list.append(brp)

if len(bme_list) == 3:
    BME = (bme_list[0] + bme_list[1] + bme_list[2]) / 3.0
    BRP = (brp_list[0] + brp_list[1] + brp_list[2]) / 3.0
    
    BME[0,:,:] = 0
    
    # Calculate absolute differences for plots
    # We will slice time, taking mean over time, and then compute |BME| - |BRP|
    def calc_plot_data(time_slice):
        mean_bme = np.abs(BME.sel(time=time_slice).mean(dim='time'))
        mean_brp = np.abs(BRP.sel(time=time_slice).mean(dim='time'))
        return mean_bme - mean_brp

    w1 = calc_plot_data(slice('2005-05-01', '2005-05-07'))
    w2 = calc_plot_data(slice('2005-05-08', '2005-05-14'))
    w34 = calc_plot_data(slice('2005-05-15', '2005-05-30'))
    w58 = calc_plot_data(slice('2005-06-01', '2005-06-30'))
    
    print("\nGenerating Robinson projection plots...")

    proj = ccrs.Robinson(central_longitude=180)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), subplot_kw={'projection': proj})

    def format_axis(ax, title):
        ax.set_global()
        ax.coastlines(linewidth=0.5, alpha=0.6)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_extent([-180, 180, -60, 85], crs=ccrs.PlateCarree())

    # Levels and colormap matching the original script preference
    # The previous instruction removed: clf=arange(-.9,.91,.2)
    # The original script for Week 1 had clf=arange(-.9,.91,.2)
    # The user asked in previous conversation:
    # "change the closeeness oclorbar, add text in the colorbar to say which color means which one is better"
    # "also make closeness plot to -5 to 5 with 1 spacing, but whiting between -1 to 1."
    # Wait, the user's latest instruction for this script says:
    # "just read the eccobias ipynb script, and recreate it everything same, except the plotting should use robinson map. again,m keep everything ekse same."
    # Let me use the PiYG setup they liked for closeness, but adapted for the values here. Or just use what they had in ECCO_bias.ipynb: RdBu_r ?
    # BME closeness was Blue, IAU closer was Red in the title "Week 1 closeness (Blue: ME Closer, Red: IAU closer)"
    # That means blue is negative, red is positive.
    
    levels = np.arange(-0.9, 0.91, 0.2)
    # Using RdBu_r => Blue for negative, Red for positive
    cmap = 'RdBu_r'

    def plot_panel(ax, da, title):
        format_axis(ax, title)
        da_cyc, lon_cyc = add_cyclic_point(da, coord=da.lon)
        p = ax.contourf(lon_cyc, da.lat, da_cyc, transform=ccrs.PlateCarree(),
                        levels=levels, cmap=cmap, extend='both')
        return p

    p1 = plot_panel(axes[0, 0], w1, 'Week 1 closeness (Blue: ME Closer, Red: IAU closer)')
    p2 = plot_panel(axes[0, 1], w2, 'Week 2 closeness (Blue: ME Closer, Red: IAU closer)')
    p3 = plot_panel(axes[1, 0], w34, 'Week 3-4 closeness (Blue: ME Closer, Red: IAU closer)')
    p4 = plot_panel(axes[1, 1], w58, 'Week 5-8 closeness (Blue: ME Closer, Red: IAU closer)')

    fig.subplots_adjust(right=0.85, wspace=0.1, hspace=0.2)
    cbar_ax = fig.add_axes([0.88, 0.25, 0.02, 0.5])
    cbar = fig.colorbar(p1, cax=cbar_ax, ticks=levels)
    cbar.set_label('Theta Bias Closeness')

    output_filename = '/Users/afahad/Library/CloudStorage/OneDrive-GeorgeMasonUniversity/MacMini/Projects/IAU_initilization/IAUinit_paper/paper_figs/theta_closeness.py_output.png'
    plt.savefig('closeness_week.png', dpi=200, bbox_inches='tight')
    plt.savefig('/nobackupp27/afahad/scripts/initialization_shock/closeness_week.png', dpi=200, bbox_inches='tight')
    print(f"Plot saved successfully.")
    
else:
    print("Missing data for 3 ensembles.")
