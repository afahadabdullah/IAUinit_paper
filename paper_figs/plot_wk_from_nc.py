import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_wk_from_file(nc_file, output_file, raw=False):
    print(f"Loading {nc_file}...")
    if not os.path.exists(nc_file):
        print(f"Error: {nc_file} not found.")
        return

    ds = xr.open_dataset(nc_file)
    power = ds['power'].values
    freqs = ds['frequency'].values
    ks = ds['zonal_wavenumber'].values
    
    if raw:
        print("Plotting RAW power (no background division)...")
        # Handle zeros for log
        # Replace 0 with small value
        power_safe = power.copy()
        power_safe[power_safe <= 0] = 1e-20
        signal = np.log10(power_safe)
        
        # Determine levels dynamically for raw power since range is unknown
        # Use percentiles to avoid outliers
        vmin = np.percentile(signal, 5)
        vmax = np.percentile(signal, 99)
        levels = np.linspace(vmin, vmax, 20)
        title_extra = "(Raw Log10 Power)"
        cmap = 'Spectral_r' # Often good for raw spectra
    else:
        print("Calculating background spectrum...")
        bg = power.copy()
        for _ in range(10):
            # Smooth in freq (axis 0)
            bg = 0.25 * np.roll(bg, 1, axis=0) + 0.5 * bg + 0.25 * np.roll(bg, -1, axis=0)
            # Smooth in wavenumber (axis 1)
            bg = 0.25 * np.roll(bg, 1, axis=1) + 0.5 * bg + 0.25 * np.roll(bg, -1, axis=1)
            
        bg[bg == 0] = 1e-20
        signal = np.log10(power / bg)
        levels = np.linspace(0, 0.6, 13)
        title_extra = "(Power/Background)"
        cmap = 'YlOrRd'
    
    # Plot
    print(f"Plotting to {output_file}...")
    plt.figure(figsize=(10, 8))
    
    plt.contourf(ks, freqs, signal, levels=levels, cmap=cmap, extend='both')
    plt.colorbar(label=f'Log10 {title_extra}')
    plt.title(f'WK Diagram {title_extra}')
    plt.xlabel('Zonal Wavenumber')
    plt.ylabel('Frequency (CPD)')
    plt.axvline(0, color='k', linestyle='--')
    plt.ylim(0.01, 0.35)
    
    # Dispersion Curves
    Re = 6.371e6; g = 9.8; beta = 2.0 * (7.2921e-5) / Re
    
    # Kelvin
    for he, c_c in zip([8, 25, 90], ['k', 'k', 'k']):
        c = np.sqrt(g * he)
        kk = np.linspace(0.1, 15, 100)
        ff = 86400.0 * c * kk / (2.0 * np.pi * Re)
        mask_p = (ff >= 0.01) & (ff <= 0.35)
        if np.any(mask_p):
            plt.plot(kk[mask_p], ff[mask_p], color=c_c, linewidth=1.5, linestyle='-')
            plt.text(kk[mask_p][-1], ff[mask_p][-1], f'{he}m', color='k', fontsize=9)
            
    # Rossby n=1
    for he, c_c in zip([8, 25, 90], ['k', 'k', 'k']):
        c = np.sqrt(g * he)
        kk_z = np.linspace(-15, -0.1, 100); kk_rad = kk_z / Re
        omega = -beta * kk_rad / (kk_rad**2 + 3 * beta / c)
        ff = 86400.0 * omega / (2.0 * np.pi)
        mask_p = (ff >= 0.01) & (ff <= 0.35)
        if np.any(mask_p):
            plt.plot(kk_z[mask_p], ff[mask_p], color=c_c, linewidth=1.5, linestyle='--')
            plt.text(kk_z[mask_p][0], ff[mask_p][0], f'{he}m', color='k', fontsize=9)
            
    plt.savefig(output_file, dpi=150)
    plt.close()
    print("Done.")

if __name__ == "__main__":
    nc_path = "/home6/afahad/nobackup27/project/initialization_shock/paper_figs/spike_wk.nc4"
    
    # 1. Standard Plot
    out_path_std = "/home6/afahad/nobackup27/project/initialization_shock/paper_figs/wk_diagram_from_nc.png"
    plot_wk_from_file(nc_path, out_path_std, raw=False)
    
    # 2. Raw Power Plot
    out_path_raw = "/home6/afahad/nobackup27/project/initialization_shock/paper_figs/wk_diagram_from_nc_raw.png"
    plot_wk_from_file(nc_path, out_path_raw, raw=True)
