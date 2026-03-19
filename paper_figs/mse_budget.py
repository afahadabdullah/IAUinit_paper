import glob
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

# ==============================================================================
# Configuration
# ==============================================================================
LAT_RANGE = (-15.0, 15.0)
LON_RANGE = (140.0, 170.0)

PT_LON, PT_LAT = 143.0, -1.0
POINT_LAT_HALF_WIDTH = 1.5
POINT_LON_HALF_WIDTH = 1.5
POINT_LAT_NEIGHBOR_COUNT = 3
POINT_LON_NEIGHBOR_COUNT = 3
FOCUS_START = "2005-05-06"
FOCUS_END = "2005-05-14"

# Choose "point" for the spike location or "box" for the regional mean.
SERIES_KIND = "point"

SPIKE_QUANTILE = 0.90
MAX_SPIKES = 3
MIN_PEAK_SEPARATION = 2
SPIKE_WINDOW_HOURS = 24
EVENT_SMOOTH_HOURS = 24
BUDGET_SMOOTH_HOURS = 48
FALLBACK_RESAMPLE_FREQ = "6h"

PROG_REQUIRED_VARS = ("T", "QV", "H")
PROG_OPTIONAL_VARS = ("DELP", "delp", "Dp", "dP", "lev_bnds", "plev_bnds", "p_bnds", "pbnds")
SURF_REQUIRED_VARS = ("PS", "SWTNET", "OLR", "SWGNET", "LWS", "LHFX", "SHFX")
SURF_OPTIONAL_VARS = (
    "PRECTOT",
    "TPREC",
    "precip",
    "LWUP",
    "LWUP_SFC",
    "LWGUP",
    "LWSUP",
    "LWGEM",
    "TS",
    "TSKIN",
    "TSA",
    "SST",
    "SST_FOUND",
    "TSURF",
    "T2M",
)

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

CACHE_DIR = Path(__file__).with_name("cache_v6")


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


def lon_distance_deg(lon_vals, target_lon):
    return np.abs(((lon_vals - target_lon + 180.0) % 360.0) - 180.0)


def format_lon_lat(lon, lat):
    lon = lon % 360.0
    lon_txt = f"{lon:.1f}E" if lon <= 180.0 else f"{360.0 - lon:.1f}W"
    lat_txt = f"{abs(lat):.1f}{'N' if lat >= 0.0 else 'S'}"
    return f"{lon_txt}, {lat_txt}"


def box_mean(da):
    weights = np.cos(np.deg2rad(da["lat"]))
    return da.weighted(weights).mean(("lat", "lon"))


def area_mean(da):
    weights = np.cos(np.deg2rad(da["lat"]))
    return da.weighted(weights).mean(("lat", "lon"))


def nearest_coord_values(coord, target, count, is_lon=False):
    values = np.asarray(coord.values, dtype=float)
    if is_lon:
        dist = lon_distance_deg(values, target)
    else:
        dist = np.abs(values - target)
    idx = np.argsort(dist)[: min(count, len(values))]
    idx = np.sort(idx)
    return values[idx]


def point_neighbor_subset(da, pt_lon=PT_LON, pt_lat=PT_LAT, lat_count=POINT_LAT_NEIGHBOR_COUNT, lon_count=POINT_LON_NEIGHBOR_COUNT):
    lon_target = canon_lon(pt_lon, da["lon"])
    lat_vals = nearest_coord_values(da["lat"], pt_lat, lat_count, is_lon=False)
    lon_vals = nearest_coord_values(da["lon"], lon_target, lon_count, is_lon=True)
    subset = da.sel(lat=lat_vals, lon=lon_vals)
    if subset.sizes.get("lat", 0) == 0 or subset.sizes.get("lon", 0) == 0:
        raise ValueError("Point-neighbor mean selection returned no grid cells.")
    return subset


def point_neighbor_metadata(da, pt_lon=PT_LON, pt_lat=PT_LAT, lat_count=POINT_LAT_NEIGHBOR_COUNT, lon_count=POINT_LON_NEIGHBOR_COUNT):
    subset = point_neighbor_subset(da, pt_lon=pt_lon, pt_lat=pt_lat, lat_count=lat_count, lon_count=lon_count)
    return {
        "lat_points_used": int(subset.sizes.get("lat", 0)),
        "lon_points_used": int(subset.sizes.get("lon", 0)),
        "grid_count_used": int(subset.sizes.get("lat", 0) * subset.sizes.get("lon", 0)),
        "lat_min_used": float(subset["lat"].min().values),
        "lat_max_used": float(subset["lat"].max().values),
        "lon_min_used": float(subset["lon"].min().values),
        "lon_max_used": float(subset["lon"].max().values),
    }


def point_neighbor_mean(da, pt_lon=PT_LON, pt_lat=PT_LAT, lat_count=POINT_LAT_NEIGHBOR_COUNT, lon_count=POINT_LON_NEIGHBOR_COUNT):
    subset = point_neighbor_subset(da, pt_lon=pt_lon, pt_lat=pt_lat, lat_count=lat_count, lon_count=lon_count)
    return area_mean(subset)


def point_box_mean(da, pt_lon=PT_LON, pt_lat=PT_LAT, lon_half_width=POINT_LON_HALF_WIDTH, lat_half_width=POINT_LAT_HALF_WIDTH):
    lat_rng = (pt_lat - lat_half_width, pt_lat + lat_half_width)
    lon_rng = (pt_lon - lon_half_width, pt_lon + lon_half_width)
    subset = sel_region(da, lat_rng, lon_rng)
    if subset.sizes.get("lat", 0) == 0 or subset.sizes.get("lon", 0) == 0:
        raise ValueError("Point-box mean selection returned no grid cells.")
    return area_mean(subset)


def require_vars(ds, required, label):
    missing = [name for name in required if name not in ds.variables]
    if missing:
        raise ValueError(f"{label} dataset is missing required variables: {missing}")


def subset_vars(ds, required, optional, label):
    require_vars(ds, required, label)
    keep = list(required) + [name for name in optional if name in ds.variables]
    return ds[keep]


def preprocess_prog(ds):
    ds = subset_vars(ds, PROG_REQUIRED_VARS, PROG_OPTIONAL_VARS, "prog")
    return sel_region(ds, LAT_RANGE, LON_RANGE)


def preprocess_surf(ds):
    ds = subset_vars(ds, SURF_REQUIRED_VARS, SURF_OPTIONAL_VARS, "surf")
    return sel_region(ds, LAT_RANGE, LON_RANGE)


def iter_files_with_progress(files, label):
    if tqdm is None:
        print(f"  -> Opening {len(files)} {label} files serially...")
        return files
    return tqdm(files, desc=f"Loading {label}", unit="file")


def load_serial_subset(file_pattern, preprocess, label, time_start="2005-04-30", time_end="2005-05-31"):
    files = sorted(glob.glob(file_pattern))
    if not files:
        raise ValueError(f"No files matched for {label}: {file_pattern}")

    pieces = []
    for path in iter_files_with_progress(files, label):
        with xr.open_dataset(path, engine="netcdf4", cache=False) as ds:
            ds = preprocess(ds)
            ds = ds.sel(time=slice(time_start, time_end))
            if ds.sizes.get("time", 0) == 0:
                continue
            pieces.append(ds.load())

    if not pieces:
        raise ValueError(f"All {label} files were empty after subsetting {time_start} to {time_end}.")

    out = xr.concat(pieces, dim="time", data_vars="minimal", coords="minimal", compat="override", combine_attrs="override")
    return out.sortby("time")


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


def resample_mean_time(ds, freq):
    sample_var = next(iter(ds.data_vars))
    counts = ds[sample_var].resample(time=freq).count()
    reduce_dims = [dim for dim in counts.dims if dim != "time"]
    if reduce_dims:
        counts = counts.sum(dim=reduce_dims)
    valid_times = counts["time"].where(counts > 0, drop=True)
    return ds.resample(time=freq).mean(keep_attrs=True).sel(time=valid_times)


def align_time_axes(state3d, flux2d, name):
    state3d = collapse_duplicate_times(state3d)
    flux2d = collapse_duplicate_times(flux2d)

    common_times = np.intersect1d(state3d.time.values, flux2d.time.values)
    if len(common_times) > 0:
        print(f"  -> Using native timestamps with {len(common_times)} common times for {name}.")
        return (
            state3d.sel(time=common_times),
            flux2d.sel(time=common_times),
            "native",
        )

    print(f"  -> No native timestamp overlap for {name}; resampling both streams to {FALLBACK_RESAMPLE_FREQ}.")
    state6 = resample_mean_time(state3d, FALLBACK_RESAMPLE_FREQ)
    flux6 = resample_mean_time(flux2d, FALLBACK_RESAMPLE_FREQ)
    common_times = np.intersect1d(state6.time.values, flux6.time.values)
    if len(common_times) == 0:
        raise ValueError(
            f"No overlapping windows found for {name} after resampling both datasets to {FALLBACK_RESAMPLE_FREQ}."
        )

    print(f"  -> Using {FALLBACK_RESAMPLE_FREQ} resampled alignment with {len(common_times)} common times.")
    return (
        state6.sel(time=common_times),
        flux6.sel(time=common_times),
        FALLBACK_RESAMPLE_FREQ,
    )


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


def shade_spike_windows(ax, time_values, spike_indices, half_window_steps):
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
        lat_neighbor_count = int(ds.attrs.get("lat_neighbor_count", POINT_LAT_NEIGHBOR_COUNT))
        lon_neighbor_count = int(ds.attrs.get("lon_neighbor_count", POINT_LON_NEIGHBOR_COUNT))
        lon_half_width = float(ds.attrs.get("lon_half_width", POINT_LON_HALF_WIDTH))
        lat_half_width = float(ds.attrs.get("lat_half_width", POINT_LAT_HALF_WIDTH))
        if "lat_neighbor_count" in ds.attrs or "lon_neighbor_count" in ds.attrs:
            return (
                f"{format_lon_lat(ds.attrs['requested_lon'], ds.attrs['requested_lat'])} "
                f"{lat_neighbor_count}x{lon_neighbor_count} neighbor mean"
            )
        return (
            f"{format_lon_lat(ds.attrs['requested_lon'], ds.attrs['requested_lat'])} "
            f"+/-{lon_half_width:.1f}deg lon, +/-{lat_half_width:.1f}deg lat"
        )
    return f"{LAT_RANGE[0]:.0f} to {LAT_RANGE[1]:.0f} lat, {LON_RANGE[0]:.0f} to {LON_RANGE[1]:.0f} lon box mean"


def print_series_averaging_info(ds, label):
    if ds.attrs.get("series_kind") != "point":
        print(f"  -> {label} uses the full lat/lon box mean over the analysis region.")
        return

    lat_points = int(ds.attrs.get("lat_points_used", ds.attrs.get("lat_neighbor_count", POINT_LAT_NEIGHBOR_COUNT)))
    lon_points = int(ds.attrs.get("lon_points_used", ds.attrs.get("lon_neighbor_count", POINT_LON_NEIGHBOR_COUNT)))
    grid_count = int(ds.attrs.get("grid_count_used", lat_points * lon_points))
    lat_min = ds.attrs.get("lat_min_used")
    lat_max = ds.attrs.get("lat_max_used")
    lon_min = ds.attrs.get("lon_min_used")
    lon_max = ds.attrs.get("lon_max_used")

    print(
        f"  -> {label} point series averages {lat_points} lat x {lon_points} lon = "
        f"{grid_count} grid cells near {format_lon_lat(ds.attrs['requested_lon'], ds.attrs['requested_lat'])}."
    )
    if None not in (lat_min, lat_max, lon_min, lon_max):
        print(
            f"     lat range used: {lat_min:.2f} to {lat_max:.2f}, "
            f"lon range used: {lon_min:.2f} to {lon_max:.2f}"
        )


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


def infer_time_step_hours(ds):
    if ds.sizes.get("time", 0) < 2:
        return 6.0
    delta = ds.time.diff("time") / np.timedelta64(1, "h")
    return float(np.nanmedian(delta.values))


def smooth_steps_from_hours(ds, target_hours=EVENT_SMOOTH_HOURS):
    dt_hours = infer_time_step_hours(ds)
    steps = max(1, int(np.ceil(target_hours / max(dt_hours, 1.0e-6))))
    if steps > 1 and steps % 2 == 0:
        steps += 1
    return steps


def rolling_event_mean(da, steps):
    if steps <= 1:
        return da
    weights = np.hanning(steps)
    if not np.isfinite(weights).all() or np.allclose(weights.sum(), 0.0):
        weights = np.ones(steps, dtype=float)
    weights = weights / weights.sum()
    weight_da = xr.DataArray(weights, dims=("window",))
    min_periods = max(1, steps // 2)
    rolled = da.rolling(time=steps, center=True, min_periods=min_periods).construct("window")
    valid = rolled.notnull()
    num = (rolled.fillna(0.0) * weight_da).sum("window")
    den = (valid.astype(float) * weight_da).sum("window")
    return xr.where(den > 0.0, num / den, np.nan)


def refine_time_for_plot(ds, spacing_hours=1):
    if ds.sizes.get("time", 0) < 2:
        return ds
    start = ds.time.values[0].astype("datetime64[h]")
    end = ds.time.values[-1].astype("datetime64[h]")
    new_time = np.arange(start, end + np.timedelta64(spacing_hours, "h"), np.timedelta64(spacing_hours, "h"))
    return ds.interp(time=new_time)


def half_window_steps_from_hours(ds, window_hours=SPIKE_WINDOW_HOURS):
    dt_hours = infer_time_step_hours(ds)
    return max(1, int(round(0.5 * window_hours / max(dt_hours, 1.0e-6))))


def integrate_flux_window(da):
    values = np.asarray(da.values, dtype=float)
    times = da.time.values.astype("datetime64[s]").astype(np.int64)
    finite = np.isfinite(values)
    if finite.sum() == 0:
        return np.nan
    values = values[finite]
    times = times[finite]
    if len(values) == 1:
        return float(values[0]) * infer_time_step_hours(da) * 3600.0
    return float(np.trapz(values, times))


def build_event_budget_table(series, spike_indices, window_hours=SPIKE_WINDOW_HOURS, smooth_hours=EVENT_SMOOTH_HOURS):
    half_steps = half_window_steps_from_hours(series, window_hours)
    smooth_col_L = rolling_event_mean(series["col_L"], smooth_steps_from_hours(series, smooth_hours))

    rows = []
    for spike_id, idx in enumerate(spike_indices, start=1):
        i0 = max(idx - half_steps, 0)
        i1 = min(idx + half_steps, series.sizes["time"] - 1)
        window = series.isel(time=slice(i0, i1 + 1))

        precip_int = integrate_flux_window(window["Precip"])
        evap_int = integrate_flux_window(window["LHF"])
        storage_int = -(
            float(smooth_col_L.isel(time=i1)) - float(smooth_col_L.isel(time=i0))
        )
        conv_int = precip_int - evap_int - storage_int
        rows.append(
            {
                "spike": f"S{spike_id}",
                "time": window.time.isel(time=idx - i0).values,
                "precip_jm2": precip_int,
                "evap_jm2": evap_int,
                "storage_jm2": storage_int,
                "conv_jm2": conv_int,
            }
        )

    return rows, half_steps


def print_event_table(rows, name):
    print(f"\n{name} event-integrated budget")
    for row in rows:
        when = np.datetime_as_string(row["time"], unit="h")
        print(
            f"  {row['spike']} {when}: "
            f"P={row['precip_jm2'] / 1.0e6:6.2f}, "
            f"E={row['evap_jm2'] / 1.0e6:6.2f}, "
            f"MC={row['conv_jm2'] / 1.0e6:6.2f}, "
            f"-dL={row['storage_jm2'] / 1.0e6:6.2f} MJ m-2"
        )


def build_event_budget_series(ds, smooth_hours=EVENT_SMOOTH_HOURS):
    steps = smooth_steps_from_hours(ds, smooth_hours)
    out = xr.Dataset(attrs=dict(ds.attrs))
    out.attrs["event_smooth_hours"] = float(smooth_hours)
    out.attrs["event_smooth_steps"] = int(steps)

    out["Precip"] = rolling_event_mean(ds["Precip"], steps)
    out["LHF"] = rolling_event_mean(ds["LHF"], steps)
    out["col_L"] = rolling_event_mean(ds["col_L"], steps)
    out["StorageRelease"] = -out["col_L"].differentiate("time", datetime_unit="s")
    out["MoistureConvergence"] = out["Precip"] - out["LHF"] - out["StorageRelease"]
    out["PrecipClosed"] = out["LHF"] + out["MoistureConvergence"] + out["StorageRelease"]
    out["PrecipResidual"] = out["Precip"] - out["PrecipClosed"]
    return out


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
    state3d = load_serial_subset(f_prog, preprocess_prog, f"{name} prog")
    flux2d = load_serial_subset(f_surf, preprocess_surf, f"{name} surf")

    if state3d.time.size == 0 or flux2d.time.size == 0:
        raise ValueError(f"Empty dataset after load for {name}. state3d={state3d.time.size}, flux2d={flux2d.time.size}")

    state3d, flux2d, align_freq = align_time_axes(state3d, flux2d, name)

    ds = xr.merge([state3d, flux2d], join="inner", compat="override")
    ds = ds.sel(time=slice(FOCUS_START, FOCUS_END))
    if ds.time.size == 0:
        raise ValueError(f"No data inside focus window {FOCUS_START} to {FOCUS_END} for {name}.")

    state3d = ds
    flux2d = ds
    print(
        f"  -> Synchronized {len(ds.time)} steps for {name} "
        f"({ds.time.min().values} to {ds.time.max().values}) using {align_freq} alignment"
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
    neighbor_meta = point_neighbor_metadata(Hnet)
    lat_rng = (PT_LAT - POINT_LAT_HALF_WIDTH, PT_LAT + POINT_LAT_HALF_WIDTH)
    lon_rng = (PT_LON - POINT_LON_HALF_WIDTH, PT_LON + POINT_LON_HALF_WIDTH)
    template = sel_region(Hnet, lat_rng, lon_rng)
    selected_lat = float(template["lat"].mean().values)
    selected_lon = float(template["lon"].mean().values)

    pt_series = make_series_dataset(
        {name_: point_neighbor_mean(da) for name_, da in fields.items()},
        {
            "series_kind": "point",
            "requested_lat": PT_LAT,
            "requested_lon": PT_LON,
            "selected_lat": selected_lat,
            "selected_lon": selected_lon,
            "lat_neighbor_count": POINT_LAT_NEIGHBOR_COUNT,
            "lon_neighbor_count": POINT_LON_NEIGHBOR_COUNT,
            "lat_half_width": POINT_LAT_HALF_WIDTH,
            "lon_half_width": POINT_LON_HALF_WIDTH,
            "point_lon_canonical": lon_c,
            "experiment_name": name,
            **neighbor_meta,
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

    print_series_averaging_info(me_series, me_series.attrs.get("experiment_name", "Reanalysis-IC"))
    print_series_averaging_info(rp_series, rp_series.attrs.get("experiment_name", "IAU-IC"))

    me_precip_plot = build_event_budget_series(me_series, smooth_hours=EVENT_SMOOTH_HOURS)
    rp_precip_plot = build_event_budget_series(rp_series, smooth_hours=EVENT_SMOOTH_HOURS)
    me_budget_plot = build_event_budget_series(me_series, smooth_hours=BUDGET_SMOOTH_HOURS)
    rp_budget_plot = build_event_budget_series(rp_series, smooth_hours=BUDGET_SMOOTH_HOURS)

    me_precip_line = refine_time_for_plot(me_precip_plot)
    rp_precip_line = refine_time_for_plot(rp_precip_plot)
    me_budget_line = refine_time_for_plot(me_budget_plot)
    rp_budget_line = refine_time_for_plot(rp_budget_plot)

    spike_indices, threshold = detect_precip_spikes(me_precip_plot["Precip"])
    half_window_steps = half_window_steps_from_hours(me_series)
    plot_label = build_plot_label(me_precip_plot)
    output_fig = Path(__file__).with_name(f"mse_budget_spike_story_{series_kind}.png")

    smooth_hours = float(me_budget_plot.attrs["event_smooth_hours"])
    print_spike_summary(
        me_budget_plot,
        spike_indices,
        (
            f"{me_budget_plot.attrs.get('experiment_name', 'Reanalysis-IC')} "
            f"({series_kind}, {EVENT_SMOOTH_HOURS:.0f}h precip / {smooth_hours:.0f}h budget smooth)"
        ),
    )
    print(f"  Spike detection threshold: {threshold:7.2f} W m-2")

    time_values = me_precip_plot.time.values
    me_line_colL_anom = (me_budget_line["col_L"] - me_budget_line["col_L"].mean("time")) * 1.0e-8
    rp_line_colL_anom = (rp_budget_line["col_L"] - rp_budget_line["col_L"].mean("time")) * 1.0e-8
    diff_precip = me_budget_line["Precip"] - rp_budget_line["Precip"]
    diff_lhf = me_budget_line["LHF"] - rp_budget_line["LHF"]
    diff_conv = me_budget_line["MoistureConvergence"] - rp_budget_line["MoistureConvergence"]
    diff_storage = me_budget_line["StorageRelease"] - rp_budget_line["StorageRelease"]
    diff_closed = me_budget_line["PrecipClosed"] - rp_budget_line["PrecipClosed"]

    fig, axes = plt.subplots(4, 1, figsize=(12, 15), sharex=True)
    fig.subplots_adjust(hspace=0.24, top=0.93)
    fig.suptitle(
        (
            f"Moisture-budget view of the precipitation spike "
            f"({plot_label}, {EVENT_SMOOTH_HOURS:.0f}h precip / {smooth_hours:.0f}h budget smooth)"
        ),
        fontsize=15,
        y=0.98,
    )

    ax = axes[0]
    shade_spike_windows(ax, time_values, spike_indices, half_window_steps)
    ax.plot(me_precip_line.time.values, me_precip_line["Precip"], color="navy", linewidth=2.8, label="Reanalysis IC")
    ax.plot(rp_precip_line.time.values, rp_precip_line["Precip"], color="darkorange", linewidth=2.3, label="IAU IC")
    ax.scatter(
        time_values[spike_indices],
        me_precip_plot["Precip"].isel(time=spike_indices),
        color="navy",
        s=34,
        zorder=5,
    )
    for n, idx in enumerate(spike_indices, start=1):
        ax.annotate(
            f"S{n}",
            (time_values[idx], float(me_precip_plot["Precip"].isel(time=idx))),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )
    ax.set_ylabel(r"$L_v P$ [W m$^{-2}$]")
    ax.set_title("(a) Precipitation spike timing", loc="left", fontweight="bold")
    ax.legend(loc="upper right", ncol=3, frameon=False, fontsize=10)

    ax = axes[1]
    shade_spike_windows(ax, time_values, spike_indices, half_window_steps)
    ax.plot(me_budget_line.time.values, me_budget_line["Precip"], color="black", linewidth=2.8, label=r"$L_v P$")
    ax.plot(me_budget_line.time.values, me_budget_line["LHF"], color="tab:blue", linewidth=2.2, label="Surface evaporation")
    ax.plot(me_budget_line.time.values, me_budget_line["MoistureConvergence"], color="tab:green", linewidth=2.2, label="Moisture convergence")
    ax.plot(me_budget_line.time.values, me_budget_line["StorageRelease"], color="tab:purple", linewidth=2.2, linestyle="--", label="Storage release")
    ax.plot(me_budget_line.time.values, me_budget_line["PrecipClosed"], color="0.45", linewidth=1.6, linestyle=":", label="E + MC + storage")
    ax.set_ylabel(r"W m$^{-2}$")
    ax.set_title("(b) Reanalysis-IC moisture source decomposition", loc="left", fontweight="bold")
    ax.legend(loc="upper right", ncol=2, frameon=False, fontsize=10)

    ax = axes[2]
    shade_spike_windows(ax, time_values, spike_indices, half_window_steps)
    ax.plot(me_budget_line.time.values, me_line_colL_anom, color="navy", linewidth=2.6, label="Reanalysis IC")
    ax.plot(rp_budget_line.time.values, rp_line_colL_anom, color="darkorange", linewidth=2.4, label="IAU IC")
    ax.axhline(0.0, color="0.4", linewidth=1.0, linestyle="--")
    ax.set_ylabel(r"$\langle L_v q \rangle'$ [$10^8$ J m$^{-2}$]")
    ax.set_title("(c) Column moisture reservoir response", loc="left", fontweight="bold")
    ax.legend(loc="upper right", frameon=False, fontsize=10)

    ax = axes[3]
    shade_spike_windows(ax, time_values, spike_indices, half_window_steps)
    ax.plot(me_budget_line.time.values, diff_precip, color="black", linewidth=2.8, label=r"$\Delta L_v P$")
    ax.plot(me_budget_line.time.values, diff_lhf, color="tab:blue", linewidth=2.2, label=r"$\Delta E$")
    ax.plot(me_budget_line.time.values, diff_conv, color="tab:green", linewidth=2.2, label=r"$\Delta$ moisture convergence")
    ax.plot(me_budget_line.time.values, diff_storage, color="tab:purple", linewidth=2.2, linestyle="--", label=r"$\Delta$ storage release")
    ax.plot(me_budget_line.time.values, diff_closed, color="0.45", linewidth=1.6, linestyle=":", label=r"$\Delta(E + MC + storage)$")
    ax.axhline(0.0, color="0.4", linewidth=1.0, linestyle="--")
    ax.set_ylabel("Reanalysis - IAU\n[W m$^{-2}$]")
    ax.set_title("(d) What makes the Reanalysis-IC spike larger?", loc="left", fontweight="bold")
    ax.legend(loc="upper right", ncol=2, frameon=False, fontsize=10)

    for ax in axes:
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.tick_params(axis="both", labelsize=10)

    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[-1].set_xlabel("Time (May 2005)")

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
