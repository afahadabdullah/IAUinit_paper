import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import dask
import os

# ==============================================================================
# Configuration
# ==============================================================================
LAT_RANGE = (-15.0, 15.0)   # Western Pacific Box
LON_RANGE = (140.0, 170.0) 

# For Point analysis (optional, replacing box mean if desired)
PT_LON, PT_LAT = 143.0, -1.0

# Paths
me_prog = '/nobackupp27/afahad/exp/IAU_exp/GEOSMIT_ME0506/holding/geosgcm_prog/200505/*prog*200505*z.nc4'
me_surf = '/nobackupp27/afahad/exp/IAU_exp/GEOSMIT_ME0506/holding/geosgcm_surf/200505/*surf*200505*z.nc4'

rp_prog = '/nobackupp27/afahad/exp/IAU_exp/GEOSMIT_RP05062/holding/geosgcm_prog/200505/*prog*200505*z.nc4'
rp_surf = '/nobackupp27/afahad/exp/IAU_exp/GEOSMIT_RP05062/holding/geosgcm_surf/200505/*surf*200505*z.nc4'

# Constants
g  = 9.81
cp = 1004.0
Lv = 2.5e6
sigma = 5.670374419e-8


# ==============================================================================
# Helper Functions
# ==============================================================================
def sel_lat(ds, lat_min=None, lat_max=None):
    if lat_min is None or lat_max is None: return ds
    la = ds["lat"]
    return ds.sel(lat=slice(lat_min, lat_max)) if la[0] < la[-1] else ds.sel(lat=slice(lat_max, lat_min))

def sel_lon(ds, lon_min=None, lon_max=None):
    if lon_min is None or lon_max is None: return ds
    lo = ds["lon"]
    if lo.max() > 180: lmin, lmax = lon_min % 360, lon_max % 360
    else:              lmin, lmax = ((lon_min + 180) % 360) - 180, ((lon_max + 180) % 360) - 180
    if lmin <= lmax:
        return ds.sel(lon=slice(lmin, lmax))
    ds1 = ds.sel(lon=slice(lmin, float(lo.max())))
    ds2 = ds.sel(lon=slice(float(lo.min()), lmax))
    return xr.concat([ds1, ds2], dim="lon")

def sel_region(ds, lat_rng=None, lon_rng=None):
    out = ds
    if lat_rng is not None: out = sel_lat(out, lat_rng[0], lat_rng[1])
    if lon_rng is not None: out = sel_lon(out, lon_rng[0], lon_rng[1])
    return out

def get_lev_dim(ds3d):
    for cand in ("lev", "plev", "level"):
        if cand in ds3d.dims: return cand
    raise ValueError("No vertical level dim found (lev/plev/level).")

def build_dp_positional(ds3d, ps_2d):
    lev_dim = get_lev_dim(ds3d)
    for key in ("DELP","delp","Dp","dP"):
        if key in ds3d:
            dp = ds3d[key]
            if lev_dim not in dp.dims: raise ValueError(f"{key} present but missing '{lev_dim}' dim.")
            return dp
    for key in ("lev_bnds","plev_bnds","p_bnds","pbnds"):
        if key in ds3d:
            pb = ds3d[key]; bdim = "bnds" if "bnds" in pb.dims else ("bound" if "bound" in pb.dims else "nbnds")
            return pb.isel({bdim:1}) - pb.isel({bdim:0})
            
    lev_vals = ds3d[lev_dim].values; nlev = int(len(lev_vals))
    p_top = 100.0  # Pa
    p_up_1d = np.empty((nlev,), dtype=float); p_dn_1d = np.empty((nlev-1,), dtype=float)
    p_up_1d[0] = p_top
    for k in range(1, nlev):   p_up_1d[k] = 0.5*(lev_vals[k-1] + lev_vals[k])
    for k in range(0, nlev-1): p_dn_1d[k] = 0.5*(lev_vals[k] + lev_vals[k+1])
    tmpl = ds3d["T"]
    p_up = xr.DataArray(p_up_1d, dims=(lev_dim,)).broadcast_like(tmpl)
    p_dn = xr.concat([
        xr.DataArray(p_dn_1d, dims=(lev_dim,)).broadcast_like(tmpl.isel({lev_dim: slice(0, nlev-1)})),
        xr.zeros_like(tmpl.isel({lev_dim: -1})) + ps_2d
    ], dim=lev_dim)
    return p_dn - p_up

def find_surface_temp(ds3d, ds2d):
    for name in ("TS","TSKIN","TSA","SST","SST_FOUND","TSURF","T2M"):
        if name in ds2d: return ds2d[name]
        if name in ds3d: return ds3d[name]
    raise ValueError("Need a surface/skin temperature for LW↑ estimate (TS/TSKIN/SST/...).")

def ensure_atm_positive(da):
    m = float(da.isel(time=0).mean().compute())
    if m < 0: return -da
    return da

# ==============================================================================
# MSE Budget Engine
# ==============================================================================
def compute_mse_budget(f_prog, f_surf, name):
    cache_dir = "cache"
    if not os.path.exists(cache_dir): os.makedirs(cache_dir)
    
    pt_file = f"{cache_dir}/{name}_pt.nc"
    box_file = f"{cache_dir}/{name}_box.nc"
    
    if os.path.exists(pt_file) and os.path.exists(box_file):
        print(f"Loading {name} from cache...")
        return xr.open_dataset(pt_file), xr.open_dataset(box_file)

    print(f"Loading {name} data from experiments...")
    # 1. Open lazily and slice time safely with a synchronous scheduler
    with dask.config.set(scheduler='single-threaded'):
        ds_prog = xr.open_mfdataset(f_prog, parallel=False).sel(time=slice('2005-05-05', '2005-05-14'))
        ds_surf = xr.open_mfdataset(f_surf, parallel=False).sel(time=slice('2005-05-05', '2005-05-14'))

        # 2. Slice spatially *before* loading into memory or doing heavy math
        state3d = sel_region(ds_prog, LAT_RANGE, LON_RANGE)
        flux2d  = sel_region(ds_surf, LAT_RANGE, LON_RANGE)

        # 3. Load the small subset into memory for fast processing
        print(f"  -> Loading spatial subset into memory...")
        state3d = state3d.load() 
        flux2d = flux2d.load()

    # 4. Resample surface fluxes to match 3D state (which is usually every 6H)
    flux2d = flux2d.resample(time='6H').mean()
    flux2d = flux2d.sel(time=state3d.time)
    
    lev_dim = get_lev_dim(state3d)
    dp = build_dp_positional(state3d, flux2d["PS"]).clip(min=0)
    
    # Mask sub-surface levels: p_mid > PS
    p_mid_vals = state3d[lev_dim].values
    p_mid_4d = xr.DataArray(p_mid_vals, dims=(lev_dim,)).broadcast_like(state3d["T"])
    below_ground = p_mid_4d > flux2d["PS"]
    
    T = state3d["T"].where(~below_ground)
    q = state3d["QV"].where(~below_ground)
    H = state3d["H"]
    Phi = g * H  

    # Column MSE (J/m^2), DSE, and Latent
    s_dry = cp*T + Phi
    q_lat = Lv*q
    h = s_dry + q_lat

    col_h = (h * dp / g).sum(lev_dim)
    col_s = (s_dry * dp / g).sum(lev_dim)
    col_L = (q_lat * dp / g).sum(lev_dim)

    # dMSEdt, dDSEdt, dLatentdt (W/m^2)
    time = col_h["time"]
    # Calculate dt in seconds correctly to avoid nanosecond scaling issues
    dt_delta = (time.shift(time=-1) - time)
    dt_sec = dt_delta.dt.total_seconds()

    def get_tendency(da):
        return (da.shift(time=-1) - da.shift(time=+1)) / (2.0*dt_sec)

    dMSEdt    = get_tendency(col_h)
    dDSEdt    = get_tendency(col_s)
    dLatentdt = get_tendency(col_L)

    # Radiation terms
    SWTNET = flux2d["SWTNET"]     
    OLR    = flux2d["OLR"]        
    SWGNET = flux2d["SWGNET"]     
    LWS_dn = flux2d["LWS"]        

    lwup_names = [n for n in ("LWUP","LWUP_SFC","LWGUP","LWSUP","LWGEM") if n in flux2d.variables]
    if lwup_names:
        LWUP_sfc = flux2d[lwup_names[0]]
    else:
        Tsurf = find_surface_temp(state3d, flux2d)  
        LWUP_sfc = sigma * Tsurf**4

    LW_net_sfc = LWS_dn - LWUP_sfc         
    SW_net_sfc = SWGNET                    
    R_sfc_net  = SW_net_sfc + LW_net_sfc   
    R_toa_net  = SWTNET - OLR              
    R_col      = R_toa_net - R_sfc_net     

    # Turbulent fluxes
    LHF = ensure_atm_positive(flux2d["LHFX"])
    SHF = ensure_atm_positive(flux2d["SHFX"])

    # Precipitation (W/m^2)
    # Common GEOS names: PRECTOT, TPREC. Multiply by Lv (J/kg) to get energy flux
    precip_var = [n for n in ("PRECTOT","TPREC","precip") if n in flux2d.variables]
    if precip_var:
        Precip = flux2d[precip_var[0]] * Lv
    else:
        Precip = xr.zeros_like(Hnet) # Placeholder if not found
        print("  WARNING: Precipitation variable not found.")

    # Heat (DSE) and Moisture (Latent) Budgets
    # dS/dt = R_col + SHF + Lv*P - DSE_export
    # dL/dt = LHF - Lv*P - Latent_export
    DSE_forcing = R_col + SHF + Precip
    Latent_forcing = LHF - Precip

    DSE_export = DSE_forcing - dDSEdt
    Latent_export = Latent_forcing - dLatentdt

    # Final terms (MSE)
    Hnet = R_col + LHF + SHF
    MSE_export = Hnet - dMSEdt

    # Point series at exact lat/lon
    def _canon_lon(val, ds_lon):
        if ds_lon.max() > 180: return val % 360.0
        return ((val + 180.0) % 360.0) - 180.0

    lon_c = _canon_lon(PT_LON, flux2d.lon)
    pt_sel = dict(lat=PT_LAT, lon=lon_c, method="nearest")

    pt_series = xr.Dataset({
        "dMSEdt": dMSEdt.sel(**pt_sel),
        "dDSEdt": dDSEdt.sel(**pt_sel),
        "dLatentdt": dLatentdt.sel(**pt_sel),
        "Hnet": Hnet.sel(**pt_sel),
        "Precip": Precip.sel(**pt_sel),
        "DSE_export": DSE_export.sel(**pt_sel),
        "Latent_export": Latent_export.sel(**pt_sel),
        "MSE_export": MSE_export.sel(**pt_sel)
    })

    # Box mean series
    wlat = np.cos(np.deg2rad(flux2d["lat"])); wlat = wlat / wlat.mean()
    def box_mean(da): return (da * wlat).mean(("lat","lon"))

    series = xr.Dataset({
        "dMSEdt": box_mean(dMSEdt),
        "dDSEdt": box_mean(dDSEdt),
        "dLatentdt": box_mean(dLatentdt),
        "Hnet": box_mean(Hnet),
        "Precip": box_mean(Precip),
        "DSE_export": box_mean(DSE_export),
        "Latent_export": box_mean(Latent_export),
        "MSE_export": box_mean(MSE_export),
    })

    print(f"  -> Caching results to {cache_dir}/...")
    pt_series.to_netcdf(pt_file)
    series.to_netcdf(box_file)

    return pt_series, series

# ==============================================================================
# Execution & Plotting
# ==============================================================================
if __name__ == '__main__':
    print("Initiating Reanalysis (ME) MSE Budget...")
    me_pt, me_box = compute_mse_budget(me_prog, me_surf, "Reanalysis-IC")
    
    print("Initiating IAU (RP) MSE Budget...")
    rp_pt, rp_box = compute_mse_budget(rp_prog, rp_surf, "IAU-IC")

    # We will plot the Point Series as requested by the user
    time_rp = rp_pt.time.values
    time_me = me_pt.time.values

    # Setup the plot - 8 Panels for Full Diagnosis
    fig, axes = plt.subplots(8, 1, figsize=(14, 25), sharex=True)
    fig.subplots_adjust(hspace=0.3)
    
    # Titles
    fig.suptitle(f'Full Energy & Moisture Budget Diagnosis at Point (Lon={PT_LON}°, Lat={PT_LAT}°)\nReanalysis vs IAU Initialization', fontsize=16)

    # 1. dMSE/dt
    ax = axes[0]
    ax.plot(time_me, me_pt['dMSEdt'], color='blue', linewidth=2.5, label='Reanalysis IC')
    ax.plot(time_rp, rp_pt['dMSEdt'], color='darkorange', linewidth=2.5, label='IAU IC')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
    ax.set_ylabel(r'$\partial\langle h \rangle/\partial t$', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title('Column Total MSE Tendency', fontsize=13)

    # 2. dDSE/dt
    ax = axes[1]
    ax.plot(time_me, me_pt['dDSEdt'], color='blue', linewidth=2)
    ax.plot(time_rp, rp_pt['dDSEdt'], color='darkorange', linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
    ax.set_ylabel(r'$\partial\langle s \rangle/\partial t$', fontsize=12)
    ax.set_title('Column Dry Static Energy Tendency (Heating/Cooling)', fontsize=13)

    # 3. dLatent/dt
    ax = axes[2]
    ax.plot(time_me, me_pt['dLatentdt'], color='blue', linewidth=2)
    ax.plot(time_rp, rp_pt['dLatentdt'], color='darkorange', linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
    ax.set_ylabel(r'$\partial\langle L_v q \rangle/\partial t$', fontsize=12)
    ax.set_title('Column Latent Energy Tendency (Moistening/Drying)', fontsize=13)

    # 4. Precipitation
    ax = axes[3]
    ax.plot(time_me, me_pt['Precip'], color='blue', linewidth=2.5)
    ax.plot(time_rp, rp_pt['Precip'], color='darkorange', linewidth=2.5)
    ax.set_ylabel(r'$L_v P$', fontsize=12)
    ax.set_title('Convective Energy Release: Precipitation ($L_v \times P$)', fontsize=13)

    # 5. DSE Export
    ax = axes[4]
    ax.plot(time_me, me_pt['DSE_export'], color='blue', linewidth=2)
    ax.plot(time_rp, rp_pt['DSE_export'], color='darkorange', linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
    ax.set_ylabel(r'DSE Export', fontsize=12)
    ax.set_title('Dynamic Heat Venting (DSE Export)', fontsize=13)

    # 6. Latent Export
    ax = axes[5]
    ax.plot(time_me, me_pt['Latent_export'], color='blue', linewidth=2)
    ax.plot(time_rp, rp_pt['Latent_export'], color='darkorange', linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
    ax.set_ylabel(r'Latent Export', fontsize=12)
    ax.set_title('Moisture Transport (Latent Export)', fontsize=13)

    # 7. Net column heating (Hnet)
    ax = axes[6]
    ax.plot(time_me, me_pt['Hnet'], color='blue', linewidth=2)
    ax.plot(time_rp, rp_pt['Hnet'], color='darkorange', linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
    ax.set_ylabel(r'$H_{net}$', fontsize=12)
    ax.set_title('Net External Forcing (Rad + Turb Flux)', fontsize=13)

    # 8. Total MSE Export
    ax = axes[7]
    ax.plot(time_me, me_pt['MSE_export'], color='blue', linewidth=2.5)
    ax.plot(time_rp, rp_pt['MSE_export'], color='darkorange', linewidth=2.5)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
    ax.set_ylabel(r'MSE Export', fontsize=12)
    ax.set_title('Total Dynamic Energy Export', fontsize=13)
    ax.set_xlabel('Time (May 2005)', fontsize=12)
    
    for a in axes:
        a.set_ylabel(a.get_ylabel() + r' [W $m^{-2}$]')
        a.grid(True, linestyle=':', alpha=0.6)

    for ax in axes:
        ax.grid(True, linestyle=':', alpha=0.6)

    output_file = 'mse_budget_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved MSE Budget comparison plot to {output_file}")
