from pathlib import Path

import dask
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# ==============================================================================
# Configuration
# ==============================================================================
LAT_RANGE = (-15.0, 15.0)
LON_RANGE = (140.0, 170.0)

PT_LON, PT_LAT = 143.0, -1.0
FOCUS_START = "2005-05-06"
FOCUS_END = "2005-05-14"

# Choose "point" for the spike location or "box" for the regional mean.
SERIES_KIND = "point"

SPIKE_QUANTILE = 0.90
MAX_SPIKES = 3
MIN_PEAK_SEPARATION = 2
SPIKE_WINDOW_STEPS = 1

# Paths
me_prog = "/nobackupp27/afahad/exp/IAU_exp/GEOSMIT_ME0506/holding/geosgcm_prog/200505/*prog*200505*z.nc4"
me_surf = "/nobackupp27/afahad/exp/IAU_exp/GEOSMIT_ME0506/holding/geosgcm_surf/200505/*surf*200505*z.nc4"

rp_prog = "/nobackupp27/afahad/exp/IAU_exp/GEOSMIT_RP05062/holding/geosgcm_prog/200505/*prog*200505*z.nc4"
rp_surf = "/nobackupp27/afahad/exp/IAU_exp/GEOSMIT_RP05062/holding/geosgcm_surf/200505/*surf*200505*z.nc4"

# Constants
g = 9.81
cp = 1004.0
Lv = 2.5e6
sigma = 5.670374419e-8

CACHE_DIR = Path(__file__).with_name("cache_v3")


# ==============================================================================
# Helper Functions
# ==============================================================================
def sel_lat(ds, lat_min=None, lat_max=None):
    if lat_min is None or lat_max is None:
        return ds
    la = ds["lat"]
    if la[0] < la[-1]:
        return ds.sel(lat=slice(lat_min, lat_max))
    return ds.sel(lat=slice(lat_max, lat_min))


def sel_lon(ds, lon_min=None, lon_max=None):
    if lon_min is None or lon_max is None:
        return ds

    lo = ds["lon"]
    if lo.max() > 180:
        lmin, lmax = lon_min % 360, lon_max % 360
    else:
        lmin = ((lon_min + 180) % 360) - 180
        lmax = ((lon_max + 180) % 360) - 180

    if lmin <= lmax:
        return ds.sel(lon=slice(lmin, lmax))

    ds1 = ds.sel(lon=slice(lmin, float(lo.max())))
    ds2 = ds.sel(lon=slice(float(lo.min()), lmax))
    return xr.concat([ds1, ds2], dim="lon")


def sel_region(ds, lat_rng=None, lon_rng=None):
    out = ds
    if lat_rng is not None:
        out = sel_lat(out, lat_rng[0], lat_rng[1])
    if lon_rng is not None:
        out = sel_lon(out, lon_rng[0], lon_rng[1])
    return out


def get_lev_dim(ds3d):
    for cand in ("lev", "plev", "level"):
        if cand in ds3d.dims:
            return cand
    raise ValueError("No vertical level dim found (lev/plev/level).")


def build_dp_positional(ds3d, ps_2d):
    lev_dim = get_lev_dim(ds3d)

    for key in ("DELP", "delp", "Dp", "dP"):
        if key in ds3d:
            dp = ds3d[key]
            if lev_dim not in dp.dims:
                raise ValueError(f"{key} present but missing '{lev_dim}' dim.")
            return dp

    for key in ("lev_bnds", "plev_bnds", "p_bnds", "pbnds"):
        if key in ds3d:
            pb = ds3d[key]
            bdim = "bnds" if "bnds" in pb.dims else ("bound" if "bound" in pb.dims else "nbnds")
            return pb.isel({bdim: 1}) - pb.isel({bdim: 0})

    lev_vals = ds3d[lev_dim].values
    nlev = int(len(lev_vals))
    p_top = 100.0
    p_up_1d = np.empty((nlev,), dtype=float)
    p_dn_1d = np.empty((nlev - 1,), dtype=float)
    p_up_1d[0] = p_top
    for k in range(1, nlev):
        p_up_1d[k] = 0.5 * (lev_vals[k - 1] + lev_vals[k])
    for k in range(0, nlev - 1):
        p_dn_1d[k] = 0.5 * (lev_vals[k] + lev_vals[k + 1])

    tmpl = ds3d["T"]
    p_up = xr.DataArray(p_up_1d, dims=(lev_dim,)).broadcast_like(tmpl)
    p_dn = xr.concat(
        [
            xr.DataArray(p_dn_1d, dims=(lev_dim,)).broadcast_like(
                tmpl.isel({lev_dim: slice(0, nlev - 1)})
            ),
            xr.zeros_like(tmpl.isel({lev_dim: -1})) + ps_2d,
        ],
        dim=lev_dim,
    )
    return p_dn - p_up


def find_surface_temp(ds3d, ds2d):
    for name in ("TS", "TSKIN", "TSA", "SST", "SST_FOUND", "TSURF", "T2M"):
        if name in ds2d:
            return ds2d[name]
        if name in ds3d:
            return ds3d[name]
    raise ValueError("Need a surface/skin temperature for LW-up estimate (TS/TSKIN/SST/...).")


def ensure_atm_positive(da):
    mean0 = float(da.isel(time=0).mean().compute())
    if mean0 < 0:
        return -da
    return da


def canon_lon(val, ds_lon):
    if ds_lon.max() > 180:
        return val % 360.0
    return ((val + 180.0) % 360.0) - 180.0


def format_lon_lat(lon, lat):
    lon = lon % 360.0
    lon_txt = f"{lon:.1f}E" if lon <= 180.0 else f"{360.0 - lon:.1f}W"
    lat_txt = f"{abs(lat):.1f}{'N' if lat >= 0.0 else 'S'}"
    return f"{lon_txt}, {lat_txt}"


def box_mean(da):
    weights = np.cos(np.deg2rad(da["lat"]))
    return da.weighted(weights).mean(("lat", "lon"))


def collapse_duplicate_times(ds):
    ds = ds.sortby("time")
    time_vals = np.asarray(ds.time.values)
    unique_times, inverse = np.unique(time_vals, return_inverse=True)
    if len(unique_times) == len(time_vals):
        return ds

    collapsed = []
    for group_id, time_value in enumerate(unique_times):
        time_idx = np.flatnonzero(inverse == group_id)
        block = ds.isel(time=time_idx)
        collapsed.append(block.mean("time", keep_attrs=True).expand_dims(time=[time_value]))

    print(f"  -> Collapsed {len(time_vals) - len(unique_times)} duplicate rounded timestamps.")
    return xr.concat(collapsed, dim="time").assign_coords(time=unique_times)


def detect_precip_spikes(precip, quantile=SPIKE_QUANTILE, min_separation=MIN_PEAK_SEPARATION, max_spikes=MAX_SPIKES):
    values = np.asarray(precip.values, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() == 0:
        return np.array([], dtype=int), np.nan

    threshold = float(np.nanquantile(values[finite], quantile))
    candidates = []
    for idx in range(len(values)):
        if not finite[idx]:
            continue
        left = values[idx - 1] if idx > 0 and np.isfinite(values[idx - 1]) else -np.inf
        right = values[idx + 1] if idx < len(values) - 1 and np.isfinite(values[idx + 1]) else -np.inf
        if values[idx] >= threshold and values[idx] >= left and values[idx] >= right:
            candidates.append(idx)

    if not candidates:
        candidates = [int(np.nanargmax(values))]

    selected = []
    for idx in sorted(candidates, key=lambda i: values[i], reverse=True):
        if all(abs(idx - old_idx) >= min_separation for old_idx in selected):
            selected.append(idx)
        if len(selected) == max_spikes:
            break

    return np.array(sorted(selected), dtype=int), threshold


def shade_spike_windows(ax, time_values, spike_indices, half_window_steps=SPIKE_WINDOW_STEPS):
    for n, idx in enumerate(spike_indices):
        left = time_values[max(idx - half_window_steps, 0)]
        right = time_values[min(idx + half_window_steps, len(time_values) - 1)]
        ax.axvspan(left, right, color="tab:red", alpha=0.10, lw=0, label="Reanalysis spike window" if n == 0 else None)
        ax.axvline(time_values[idx], color="tab:red", linestyle="--", linewidth=1.0, alpha=0.7)


def make_series_dataset(fields, metadata):
    ds = xr.Dataset(fields)
    ds.attrs.update(metadata)
    return ds


def build_plot_label(ds):
    if ds.attrs.get("series_kind") == "point":
        return format_lon_lat(ds.attrs["selected_lon"], ds.attrs["selected_lat"])
    return f"{LAT_RANGE[0]:.0f} to {LAT_RANGE[1]:.0f} lat, {LON_RANGE[0]:.0f} to {LON_RANGE[1]:.0f} lon box mean"


def print_spike_summary(series, spike_indices, name):
    print(f"\n{name} spike summary")
    for spike_id, idx in enumerate(spike_indices, start=1):
        when = np.datetime_as_string(series.time.isel(time=idx).values, unit="h")
        precip = float(series["Precip"].isel(time=idx))
        evap = float(series["LHF"].isel(time=idx))
        conv = float(series["MoistureConvergence"].isel(time=idx))
        storage = float(series["StorageRelease"].isel(time=idx))
        residual = float(series["PrecipResidual"].isel(time=idx))

        if abs(precip) > 1.0e-12:
            frac_e = 100.0 * evap / precip
            frac_c = 100.0 * conv / precip
            frac_s = 100.0 * storage / precip
        else:
            frac_e = frac_c = frac_s = np.nan

        contributions = {
            "surface evaporation": evap,
            "moisture convergence": conv,
            "storage release": storage,
        }
        dominant = max(contributions, key=lambda key: abs(contributions[key]))
        print(
            f"  S{spike_id} {when}: LvP={precip:7.2f}, "
            f"E={evap:7.2f} ({frac_e:5.1f}%), "
            f"MC={conv:7.2f} ({frac_c:5.1f}%), "
            f"storage={storage:7.2f} ({frac_s:5.1f}%), "
            f"residual={residual:8.3f}, dominant={dominant}"
        )


# ==============================================================================
# MSE Budget Engine
# ==============================================================================
def compute_mse_budget(f_prog, f_surf, name):
    CACHE_DIR.mkdir(exist_ok=True)
    pt_file = CACHE_DIR / f"{name}_pt.nc"
    box_file = CACHE_DIR / f"{name}_box.nc"

    if pt_file.exists() and box_file.exists():
        print(f"Loading {name} from cache...")
        return xr.open_dataset(pt_file).load(), xr.open_dataset(box_file).load()

    print(f"Loading {name} data from experiments...")
    with dask.config.set(scheduler="single-threaded"):
        ds_prog = xr.open_mfdataset(f_prog, parallel=False).sel(time=slice("2005-04-30", "2005-05-31"))
        ds_surf = xr.open_mfdataset(f_surf, parallel=False).sel(time=slice("2005-04-30", "2005-05-31"))

        state3d = sel_region(ds_prog, LAT_RANGE, LON_RANGE).load()
        flux2d = sel_region(ds_surf, LAT_RANGE, LON_RANGE).load()

    if state3d.time.size == 0 or flux2d.time.size == 0:
        raise ValueError(f"Empty dataset after load for {name}. state3d={state3d.time.size}, flux2d={flux2d.time.size}")

    print(f"  -> Snapping {name} timestamps to nearest 6h and aligning...")
    state3d["time"] = state3d.time.dt.round("6h")
    flux2d["time"] = flux2d.time.dt.round("6h")

    state3d = collapse_duplicate_times(state3d)
    flux2d = collapse_duplicate_times(flux2d)

    common_times = np.intersect1d(state3d.time, flux2d.time)
    if len(common_times) == 0:
        raise ValueError(f"No overlapping 6-hour windows found for {name}.")

    state3d = state3d.sel(time=common_times)
    flux2d = flux2d.sel(time=common_times)

    ds = xr.merge([state3d, flux2d], join="inner", compat="override")
    ds = ds.sel(time=slice(FOCUS_START, FOCUS_END))
    if ds.time.size == 0:
        raise ValueError(f"No data inside focus window {FOCUS_START} to {FOCUS_END} for {name}.")

    state3d = ds
    flux2d = ds
    print(
        f"  -> Synchronized {len(ds.time)} steps for {name} "
        f"({ds.time.min().values} to {ds.time.max().values})"
    )

    lev_dim = get_lev_dim(state3d)
    has_delp = any(n in state3d for n in ("DELP", "delp", "Dp", "dP"))
    p_coord = state3d[lev_dim]
    lev_is_hpa = bool(p_coord.max() > 100 and p_coord.max() < 2000)

    dp = build_dp_positional(state3d, flux2d["PS"]).clip(min=0)
    if not has_delp and lev_is_hpa:
        dp = dp * 100.0

    p_mid_vals = state3d[lev_dim].values
    if lev_is_hpa:
        p_mid_vals = p_mid_vals * 100.0
    p_mid_4d = xr.DataArray(p_mid_vals, dims=(lev_dim,)).broadcast_like(state3d["T"])
    below_ground = p_mid_4d > flux2d["PS"]

    T = state3d["T"].where(~below_ground)
    q = state3d["QV"].where(~below_ground)
    Phi = g * state3d["H"].where(~below_ground)

    q_lat = Lv * q
    col_L = (q_lat * dp / g).sum(lev_dim, skipna=True)
    s_dry = cp * T + Phi
    col_s = (s_dry * dp / g).sum(lev_dim, skipna=True)
    col_h = col_s + col_L

    dMSEdt = col_h.differentiate("time", datetime_unit="s")
    dDSEdt = col_s.differentiate("time", datetime_unit="s")
    dLatentdt = col_L.differentiate("time", datetime_unit="s")

    SWTNET = flux2d["SWTNET"]
    OLR = flux2d["OLR"]
    SWGNET = flux2d["SWGNET"]
    LWS_dn = flux2d["LWS"]

    lwup_names = [n for n in ("LWUP", "LWUP_SFC", "LWGUP", "LWSUP", "LWGEM") if n in flux2d.variables]
    if lwup_names:
        LWUP_sfc = flux2d[lwup_names[0]]
    else:
        Tsurf = find_surface_temp(state3d, flux2d)
        LWUP_sfc = sigma * Tsurf**4

    LW_net_sfc = LWS_dn - LWUP_sfc
    SW_net_sfc = SWGNET
    R_net_sfc = SW_net_sfc + LW_net_sfc
    R_net_toa = SWTNET - OLR
    R_col = R_net_toa - R_net_sfc

    LHF = ensure_atm_positive(flux2d["LHFX"])
    SHF = ensure_atm_positive(flux2d["SHFX"])

    precip_var = [n for n in ("PRECTOT", "TPREC", "precip") if n in flux2d.variables]
    if precip_var:
        Precip = flux2d[precip_var[0]] * Lv
    else:
        Precip = xr.zeros_like(R_col)
        print("  WARNING: precipitation variable not found.")

    DSE_forcing = R_col + SHF + Precip
    Latent_forcing = LHF - Precip

    DSE_export = DSE_forcing - dDSEdt
    Latent_export = Latent_forcing - dLatentdt
    Hnet = R_col + LHF + SHF
    MSE_export = Hnet - dMSEdt

    MoistureConvergence = -Latent_export
    StorageRelease = -dLatentdt
    PrecipClosed = LHF + MoistureConvergence + StorageRelease
    PrecipResidual = Precip - PrecipClosed

    fields = {
        "dMSEdt": dMSEdt,
        "dDSEdt": dDSEdt,
        "dLatentdt": dLatentdt,
        "col_L": col_L,
        "col_s": col_s,
        "col_h": col_h,
        "R_col": R_col,
        "LHF": LHF,
        "SHF": SHF,
        "Hnet": Hnet,
        "Precip": Precip,
        "DSE_export": DSE_export,
        "Latent_export": Latent_export,
        "MSE_export": MSE_export,
        "MoistureConvergence": MoistureConvergence,
        "StorageRelease": StorageRelease,
        "PrecipClosed": PrecipClosed,
        "PrecipResidual": PrecipResidual,
    }

    lon_c = canon_lon(PT_LON, flux2d.lon)
    pt_sel = dict(lat=PT_LAT, lon=lon_c, method="nearest")
    template = Hnet.sel(**pt_sel)
    selected_lat = float(template["lat"].values)
    selected_lon = float(template["lon"].values)

    pt_series = make_series_dataset(
        {name_: da.sel(**pt_sel) for name_, da in fields.items()},
        {
            "series_kind": "point",
            "requested_lat": PT_LAT,
            "requested_lon": PT_LON,
            "selected_lat": selected_lat,
            "selected_lon": selected_lon,
            "experiment_name": name,
        },
    )

    box_series = make_series_dataset(
        {name_: box_mean(da) for name_, da in fields.items()},
        {
            "series_kind": "box",
            "lat_min": LAT_RANGE[0],
            "lat_max": LAT_RANGE[1],
            "lon_min": LON_RANGE[0],
            "lon_max": LON_RANGE[1],
            "experiment_name": name,
        },
    )

    print(f"  -> Caching results to {CACHE_DIR}/...")
    pt_series.to_netcdf(pt_file)
    box_series.to_netcdf(box_file)

    return pt_series, box_series


# ==============================================================================
# Plotting
# ==============================================================================
def plot_spike_budget(me_series, rp_series, series_kind):
    if series_kind not in ("point", "box"):
        raise ValueError("SERIES_KIND must be 'point' or 'box'.")

    me_series, rp_series = xr.align(me_series, rp_series, join="inner")
    if me_series.time.size == 0:
        raise ValueError("No common time points after aligning the two experiments.")

    spike_indices, threshold = detect_precip_spikes(me_series["Precip"])
    plot_label = build_plot_label(me_series)
    output_fig = Path(__file__).with_name(f"mse_budget_spike_story_{series_kind}.png")

    print_spike_summary(me_series, spike_indices, f"{me_series.attrs.get('experiment_name', 'Reanalysis-IC')} ({series_kind})")
    print(f"  Spike detection threshold: {threshold:7.2f} W m-2")

    time_values = me_series.time.values
    me_colL_anom = (me_series["col_L"] - me_series["col_L"].mean("time")) * 1.0e-8
    rp_colL_anom = (rp_series["col_L"] - rp_series["col_L"].mean("time")) * 1.0e-8

    diff_precip = me_series["Precip"] - rp_series["Precip"]
    diff_lhf = me_series["LHF"] - rp_series["LHF"]
    diff_conv = me_series["MoistureConvergence"] - rp_series["MoistureConvergence"]
    diff_storage = me_series["StorageRelease"] - rp_series["StorageRelease"]
    diff_closed = diff_lhf + diff_conv + diff_storage

    fig, axes = plt.subplots(4, 1, figsize=(12, 15), sharex=True)
    fig.subplots_adjust(hspace=0.24, top=0.93)
    fig.suptitle(
        f"Moisture-budget view of the precipitation spike ({plot_label})",
        fontsize=15,
        y=0.98,
    )

    ax = axes[0]
    shade_spike_windows(ax, time_values, spike_indices)
    ax.plot(time_values, me_series["Precip"], color="navy", linewidth=2.8, label="Reanalysis IC")
    ax.plot(time_values, rp_series["Precip"], color="darkorange", linewidth=2.3, label="IAU IC")
    ax.scatter(
        time_values[spike_indices],
        me_series["Precip"].isel(time=spike_indices),
        color="navy",
        s=34,
        zorder=5,
    )
    for n, idx in enumerate(spike_indices, start=1):
        ax.annotate(
            f"S{n}",
            (time_values[idx], float(me_series["Precip"].isel(time=idx))),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )
    ax.set_ylabel(r"$L_v P$ [W m$^{-2}$]")
    ax.set_title("(a) Precipitation spike timing", loc="left", fontweight="bold")
    ax.legend(loc="upper right", ncol=3, frameon=False, fontsize=10)

    ax = axes[1]
    shade_spike_windows(ax, time_values, spike_indices)
    ax.plot(time_values, me_series["Precip"], color="black", linewidth=2.8, label=r"$L_v P$")
    ax.plot(time_values, me_series["LHF"], color="tab:blue", linewidth=2.2, label="Surface evaporation")
    ax.plot(time_values, me_series["MoistureConvergence"], color="tab:green", linewidth=2.2, label="Moisture convergence")
    ax.plot(time_values, me_series["StorageRelease"], color="tab:purple", linewidth=2.2, linestyle="--", label="Storage release")
    ax.plot(time_values, me_series["PrecipClosed"], color="0.45", linewidth=1.6, linestyle=":", label="E + MC + storage")
    ax.set_ylabel(r"W m$^{-2}$")
    ax.set_title("(b) Reanalysis-IC moisture source decomposition", loc="left", fontweight="bold")
    ax.legend(loc="upper right", ncol=2, frameon=False, fontsize=10)

    ax = axes[2]
    shade_spike_windows(ax, time_values, spike_indices)
    ax.plot(time_values, me_colL_anom, color="navy", linewidth=2.6, label="Reanalysis IC")
    ax.plot(time_values, rp_colL_anom, color="darkorange", linewidth=2.4, label="IAU IC")
    ax.axhline(0.0, color="0.4", linewidth=1.0, linestyle="--")
    ax.set_ylabel(r"$\langle L_v q \rangle'$ [$10^8$ J m$^{-2}$]")
    ax.set_title("(c) Column moisture reservoir response", loc="left", fontweight="bold")
    ax.legend(loc="upper right", frameon=False, fontsize=10)

    ax = axes[3]
    shade_spike_windows(ax, time_values, spike_indices)
    ax.plot(time_values, diff_precip, color="black", linewidth=2.8, label=r"$\Delta L_v P$")
    ax.plot(time_values, diff_lhf, color="tab:blue", linewidth=2.2, label=r"$\Delta E$")
    ax.plot(time_values, diff_conv, color="tab:green", linewidth=2.2, label=r"$\Delta$ moisture convergence")
    ax.plot(time_values, diff_storage, color="tab:purple", linewidth=2.2, linestyle="--", label=r"$\Delta$ storage release")
    ax.plot(time_values, diff_closed, color="0.45", linewidth=1.6, linestyle=":", label=r"$\Delta(E + MC + storage)$")
    ax.axhline(0.0, color="0.4", linewidth=1.0, linestyle="--")
    ax.set_ylabel("Reanalysis - IAU\n[W m$^{-2}$]")
    ax.set_title("(d) What makes the Reanalysis-IC spike larger?", loc="left", fontweight="bold")
    ax.legend(loc="upper right", ncol=2, frameon=False, fontsize=10)

    for ax in axes:
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.tick_params(axis="both", labelsize=10)

    axes[-1].set_xlabel("Time (May 2005)")
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    plt.savefig(output_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {output_fig}")


# ==============================================================================
# Execution
# ==============================================================================
if __name__ == "__main__":
    print("Initiating Reanalysis (ME) MSE budget...")
    me_pt, me_box = compute_mse_budget(me_prog, me_surf, "Reanalysis-IC")

    print("Initiating IAU (RP) MSE budget...")
    rp_pt, rp_box = compute_mse_budget(rp_prog, rp_surf, "IAU-IC")

    selected = {
        "point": (me_pt, rp_pt),
        "box": (me_box, rp_box),
    }
    if SERIES_KIND not in selected:
        raise ValueError("SERIES_KIND must be either 'point' or 'box'.")
    me_series, rp_series = selected[SERIES_KIND]
    plot_spike_budget(me_series, rp_series, SERIES_KIND)
