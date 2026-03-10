import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
import sys

"""
WK Analysis for Spike Variables (ME506, ME416, Ensemble Spectrum)
"""

output_dir = "/home6/afahad/.gemini/antigravity/brain/e1543863-87a1-4b5f-92e5-ef8fb0e64022"

def load_and_prep_file(file_path, var_name='sp'):
    print(f"Loading {file_path}...")
    try:
        ds = xr.open_dataset(file_path)
        if var_name in ds:
            da = ds[var_name]
            if da.dims != ('time', 'lat', 'lon'):
                if 'time' in da.dims and 'lat' in da.dims and 'lon' in da.dims:
                    da = da.transpose('time', 'lat', 'lon')
                else:
                    raise ValueError(f"Dimensions mismatch: {da.dims}")
        else:
            raise ValueError(f"Variable {var_name} not found.")
            
        if da.lon.min() < 0:
            da.coords['lon'] = (da.coords['lon'] + 360) % 360
            da = da.sortby('lon')
            
        da = da.sortby('time')
        da = da.fillna(0.0)
        return da
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def plot_wk_diagram(power_subset, freqs_sub, ks_sub, label_id, output_path):
    """
    Helper to plot WK diagram given power spectrum subset.
    """
    # Background Smoothing
    bg = power_subset.copy()
    for _ in range(10):
        bg = 0.25 * np.roll(bg, 1, axis=0) + 0.5 * bg + 0.25 * np.roll(bg, -1, axis=0)
        bg = 0.25 * np.roll(bg, 1, axis=1) + 0.5 * bg + 0.25 * np.roll(bg, -1, axis=1)
        
    signal = np.log10(power_subset / bg)
    
    plt.figure(figsize=(10, 8))
    levels = np.linspace(0, 0.6, 13)
    plt.contourf(ks_sub, freqs_sub, signal, levels=levels, cmap='YlOrRd', extend='max')
    plt.colorbar(label='Log10(Power/Background)')
    plt.title(f'WK Diagram - {label_id}')
    plt.xlabel('Zonal Wavenumber')
    plt.ylabel('Frequency (CPD)')
    plt.axvline(0, color='k', linestyle='--')
    plt.ylim(0.01, 0.35)
    
    # Dispersion Curves
    Re = 6.371e6; g = 9.8; beta = 2.0 * (7.2921e-5) / Re
    
    for he, c_c in zip([8, 25, 90], ['k', 'k', 'k']):
        c = np.sqrt(g * he)
        kk = np.linspace(0.1, 15, 100)
        ff = 86400.0 * c * kk / (2.0 * np.pi * Re)
        mask_p = (ff >= 0.01) & (ff <= 0.35)
        if np.any(mask_p):
            plt.plot(kk[mask_p], ff[mask_p], color=c_c, linewidth=1.5, linestyle='-')
            plt.text(kk[mask_p][-1], ff[mask_p][-1], f'{he}m', color='k', fontsize=9)
            
    for he, c_c in zip([8, 25, 90], ['k', 'k', 'k']):
        c = np.sqrt(g * he)
        kk_z = np.linspace(-15, -0.1, 100); kk_rad = kk_z / Re
        omega = -beta * kk_rad / (kk_rad**2 + 3 * beta / c)
        ff = 86400.0 * omega / (2.0 * np.pi)
        mask_p = (ff >= 0.01) & (ff <= 0.35)
        if np.any(mask_p):
            plt.plot(kk_z[mask_p], ff[mask_p], color=c_c, linewidth=1.5, linestyle='--')
            plt.text(kk_z[mask_p][0], ff[mask_p][0], f'{he}m', color='k', fontsize=9)
            
    plt.savefig(output_path, dpi=150)
    plt.close()

def analyze_case(da_input, label_id):
    print(f"\n==========================================")
    print(f"Starting Analysis for: {label_id}")
    print(f"==========================================")
    
    lat_min, lat_max = -15, 15
    data_trop = da_input.sel(lat=slice(lat_min, lat_max))
    anom = data_trop - data_trop.mean(dim='time')
    
    if 0 not in anom.lat:
        new_lats = np.arange(lat_min, lat_max + 1.0, 1.0)
        anom = anom.interp(lat=new_lats, method='linear')
        
    nlat = len(anom.lat)
    data_sym = np.zeros_like(anom)
    for i in range(nlat):
        l = anom.lat.values[i]
        try:
            val_pos = anom.sel(lat=l, method='nearest')
            val_neg = anom.sel(lat=-l, method='nearest')
            data_sym[:, i, :] = 0.5 * (val_pos + val_neg)
        except:
            pass
            
    component = xr.DataArray(data_sym, coords=anom.coords, dims=anom.dims)
    
    nt, nlat, nlon = component.shape
    taper = np.hanning(nt)
    component_tapered = component * taper[:, None, None]
    
    print("Computing FFT...")
    fft_sym = np.fft.fft2(component_tapered.values, axes=(0, 2))
    power = np.abs(fft_sym)**2
    power_lat_sum = np.sum(power, axis=1)
    power_shifted = np.fft.fftshift(power_lat_sum, axes=1)
    
    freqs = np.fft.fftfreq(nt, d=0.125)
    ks = np.fft.fftshift(np.fft.fftfreq(nlon, d=1.0/nlon))
    
    freq_mask = (freqs > 0) & (freqs < 0.8)
    k_mask = (ks >= -15) & (ks <= 15)
    
    freq_idxs = np.where(freq_mask)[0]
    k_idxs = np.where(k_mask)[0]
    
    power_subset = power_shifted[freq_idxs[:, None], k_idxs]
    freqs_sub = freqs[freq_idxs]
    ks_sub = ks[k_idxs]
    
    # Plot Individual WK
    plot_wk_diagram(power_subset, freqs_sub, ks_sub, label_id, 
                   os.path.join(output_dir, f'wk_diagram_spike_{label_id}.png'))
    
    # ----------------------------
    # Filtering & Point Analysis (Run for individual members)
    # ----------------------------
    fft_raw = np.fft.fft2(component.values, axes=(0, 2))
    
    start_unshift = np.fft.fftfreq(nt, d=0.125)
    ks_unshift = np.fft.fftfreq(nlon, d=1.0/nlon)
    
    filt_kel_2d = np.zeros((nt, nlon), dtype=bool)
    filt_mjo_2d = np.zeros((nt, nlon), dtype=bool)
    
    factor = 86400.0 / (2.0 * np.pi * 6.371e6); g = 9.8
    c_min = np.sqrt(g*8); c_max = np.sqrt(g*90)
    
    for t in range(nt):
        f = start_unshift[t]
        for x in range(nlon):
            k = ks_unshift[x]
            if (f > 1/30.0) and (f < 1/2.5) and (k >= 1) and (k <= 14):
                f_c = k * factor
                if (f >= c_min * f_c) and (f <= c_max * f_c): filt_kel_2d[t, x] = True
            elif (f < -1/30.0) and (f > -1/2.5) and (k <= -1) and (k >= -14):
                 # Symmetry
                 if (f/k/factor >= c_min) and (f/k/factor <= c_max): filt_kel_2d[t, x] = True
            
            if (abs(f) > 1/96.0) and (abs(f) < 1/30.0) and (abs(k) >= 1) and (abs(k) <= 5):
                filt_mjo_2d[t, x] = True
                
    mask_kel = np.zeros_like(fft_raw, dtype=bool); mask_kel[:] = filt_kel_2d[:, None, :]
    mask_mjo = np.zeros_like(fft_raw, dtype=bool); mask_mjo[:] = filt_mjo_2d[:, None, :]
    
    kelvin_data = np.fft.ifft2(fft_raw * mask_kel, axes=(0,2)).real
    mjo_data = np.fft.ifft2(fft_raw * mask_mjo, axes=(0,2)).real
    
    da_kel = xr.DataArray(kelvin_data, coords=component.coords, dims=component.dims)
    da_mjo = xr.DataArray(mjo_data, coords=component.coords, dims=component.dims)
    
    points = [
        {'label': f'Region 1 (143E, -1N)', 'lon': 143, 'lat': -1},
        {'label': f'Region 2 (215E, 8N)',  'lon': 215, 'lat': 8},
        {'label': f'Region 3 (233E, 7N)',  'lon': 233, 'lat': 7}
    ]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    for i, pt in enumerate(points):
        ax = axes[i]
        ts_anom = anom.sel(lon=pt['lon'], lat=pt['lat'], method='nearest')
        ts_kel = da_kel.sel(lon=pt['lon'], lat=pt['lat'], method='nearest')
        ts_mjo = da_mjo.sel(lon=pt['lon'], lat=pt['lat'], method='nearest')
        
        ts_anom.plot(ax=ax, label='Anom', color='k', alpha=0.6)
        ts_kel.plot(ax=ax, label='Kelvin', color='r', linewidth=2)
        ts_mjo.plot(ax=ax, label='MJO', color='b', linestyle='--', linewidth=2)
        ax.set_title(f"{pt['label']} - {label_id}")
        ax.set_ylabel('Count')
        if i==0: ax.legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'point_timeseries_spike_{label_id}.png'), dpi=150)
    plt.close()
    
    return power_subset, freqs_sub, ks_sub

if __name__ == "__main__":
    file_506 = "/home6/afahad/nobackup27/project/initialization_shock/paper_figs/spike_ME506.nc"
    file_416 = "/home6/afahad/nobackup27/project/initialization_shock/paper_figs/spike_ME416.nc"
    
    # Store results
    res_506 = None
    res_416 = None
    
    # Run ME506
    da_506 = load_and_prep_file(file_506)
    if da_506 is not None:
        res_506 = analyze_case(da_506, "ME506")
        
    # Run ME416
    da_416 = load_and_prep_file(file_416)
    if da_416 is not None:
        res_416 = analyze_case(da_416, "ME416")
        
    # Compute Mean Spectrum if both exist
    if res_506 is not None and res_416 is not None:
        print("\nCalculating Ensemble Mean Spectrum...")
        p506, f506, k506 = res_506
        p416, f416, k416 = res_416
        
        # Assume coords match (same grid/time)
        p_ens = (p506 + p416) / 2.0
        
        plot_wk_diagram(p_ens, f506, k506, "Ensemble_Spectrum",
                       os.path.join(output_dir, 'wk_diagram_spike_Ensemble.png'))
        print("Ensemble Mean Spectrum WK diagram saved.")

        # Save to NetCDF as requested
        save_path = "/home6/afahad/nobackup27/project/initialization_shock/paper_figs/spike_wk.nc4"
        print(f"Saving ensemble WK data to {save_path}...")
        try:
            ds_out = xr.Dataset(
                data_vars=dict(
                    power=(["frequency", "zonal_wavenumber"], p_ens)
                ),
                coords=dict(
                    frequency=f506,
                    zonal_wavenumber=k506
                ),
                attrs=dict(description="Ensemble Mean (ME506+ME416) WK Power Spectrum for Spike variable")
            )
            ds_out.to_netcdf(save_path)
            print("File saved successfully.")
        except Exception as e:
            print(f"Error saving NetCDF: {e}")
