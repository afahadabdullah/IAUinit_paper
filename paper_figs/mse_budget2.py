"""Western Tropical Pacific moisture-budget test with direct moisture convergence.

This diagnostic is intentionally separate from spike_budget_stats.py. It loads a
small point-centered stencil synchronously, computes vertically integrated
horizontal moisture-flux convergence from q, u, and v, then samples the nearest
WTP grid cell.
"""

import csv
import glob
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import xarray as xr

import mse_budget as mb


ANALYSIS_START = "2005-05-01"
ANALYSIS_END = "2005-06-30"
SPIKE_QUANTILE = 0.95
MIN_PEAK_SEPARATION_HOURS = 24.0
EVENT_WINDOW_HOURS = 24.0
PRECIP_SMOOTH_HOURS = 6.0
SECONDS_PER_DAY = 86400.0
EARTH_RADIUS_M = 6_371_000.0

WTP_TARGET = {"key": "wtp", "title": "Western Tropical Pacific", "lon": (143.0, 143.0), "lat": (-1.0, -1.0)}
POINT_STENCIL_COUNT = 3

CACHE_DIR = Path(__file__).with_name("cache_mse_budget2_direct_mc_wtp_point_sync")
EVENT_TABLE_PATH = Path(__file__).with_name("mse_budget2_wtp_direct_mc_events.csv")
SUMMARY_PATH = Path(__file__).with_name("mse_budget2_wtp_direct_mc_summary.csv")
MEAN_FIGURE_PATH = Path(__file__).with_name("mse_budget2_wtp_direct_mc_budget.png")
STORY_FIGURE_PATH = Path(__file__).with_name("mse_budget2_wtp_direct_mc_story.png")

PLOT_COMPONENTS = [
    ("precip_mm", r"$\int P\,dt$"),
    ("evap_mm", r"$\int E\,dt$"),
    ("mc_direct_mm", r"$\int MC_{\mathrm{direct}}\,dt$"),
    ("storage_mm", r"$-\Delta W$"),
]

U_CANDIDATES = ("U", "UWND", "UGRD", "ua")
V_CANDIDATES = ("V", "VWND", "VGRD", "va")
PROG_KEEP = (
    "QV",
    *U_CANDIDATES,
    *V_CANDIDATES,
    "DELP",
    "delp",
    "Dp",
    "dP",
    "lev_bnds",
    "plev_bnds",
    "p_bnds",
    "pbnds",
)
SURF_KEEP = ("PS", "LHFX", "PRECTOT", "TPREC", "precip")


def monthly_patterns(pattern, months=("200505", "200506")):
    return [pattern.replace("200505", month) for month in months]


def expand_files(patterns):
    files = []
    for pattern in patterns:
        files.extend(sorted(glob.glob(pattern)))
    unique = []
    seen = set()
    for path in files:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def first_existing(ds, names, label):
    for name in names:
        if name in ds:
            return name
    raise ValueError(f"No {label} variable found. Tried: {names}. Available: {list(ds.data_vars)}")


def subset_existing(ds, keep_names, required, label):
    missing = [name for name in required if name not in ds]
    if missing:
        raise ValueError(f"{label} dataset is missing required variables: {missing}")
    keep = [name for name in keep_names if name in ds]
    return ds[keep]


def point_from_region(region):
    return (
        0.5 * (region["lat"][0] + region["lat"][1]),
        0.5 * (region["lon"][0] + region["lon"][1]),
    )


def select_nearest_stencil(ds, region, count=POINT_STENCIL_COUNT):
    lat0, lon0 = point_from_region(region)
    lat_vals = mb.nearest_coord_values(ds["lat"], lat0, count, is_lon=False)
    lon_target = mb.canon_lon(lon0, ds["lon"])
    lon_vals = mb.nearest_coord_values(ds["lon"], lon_target, count, is_lon=True)
    return ds.sel(lat=lat_vals, lon=lon_vals)


def select_nearest_point(ds, region):
    lat0, lon0 = point_from_region(region)
    lat_vals = mb.nearest_coord_values(ds["lat"], lat0, 1, is_lon=False)
    lon_target = mb.canon_lon(lon0, ds["lon"])
    lon_vals = mb.nearest_coord_values(ds["lon"], lon_target, 1, is_lon=True)
    return ds.sel(lat=lat_vals, lon=lon_vals)


def preprocess_prog(region):
    def _preprocess(ds):
        ds = subset_existing(ds, PROG_KEEP, ("QV",), "prog")
        first_existing(ds, U_CANDIDATES, "zonal wind")
        first_existing(ds, V_CANDIDATES, "meridional wind")
        return select_nearest_stencil(ds, region)

    return _preprocess


def preprocess_surf(region):
    def _preprocess(ds):
        ds = subset_existing(ds, SURF_KEEP, ("PS", "LHFX"), "surf")
        return select_nearest_stencil(ds, region)

    return _preprocess


def describe_source_files(patterns, keep_names, label):
    files = expand_files(patterns)
    print(f"\n{label} source diagnostics")
    print(f"  matched files: {len(files)}")
    if not files:
        raise ValueError(f"No files matched for {label}: {patterns}")

    with xr.open_dataset(files[0], engine="netcdf4", cache=False) as ds:
        loaded = [name for name in keep_names if name in ds]
        print(f"  first file: {files[0]}")
        print(f"  variables requested/found: {loaded}")
        for name in loaded:
            da = ds[name]
            units = da.attrs.get("units", "unknown")
            long_name = da.attrs.get("long_name", da.attrs.get("standard_name", ""))
            print(f"    {name}: dims={da.dims}, shape={da.shape}, units={units}, {long_name}")
        print(f"  coordinates: {[name for name in ('time', 'lev', 'plev', 'level', 'lat', 'lon') if name in ds]}")
    return files


def load_sync_stencil_patterns(patterns, preprocess, label):
    files = expand_files(patterns)
    if not files:
        raise ValueError(f"No files matched for {label}: {patterns}")

    print(f"  -> Opening {len(files)} {label} files synchronously with point stencil.")
    pieces = []
    for path in mb.iter_files_with_progress(files, label):
        with xr.open_dataset(path, engine="netcdf4", cache=False) as ds:
            ds = preprocess(ds)
            ds = ds.sel(time=slice(ANALYSIS_START, ANALYSIS_END))
            if ds.sizes.get("time", 0) == 0:
                continue
            pieces.append(ds.load())

    if not pieces:
        raise ValueError(f"All {label} files were empty after subsetting {ANALYSIS_START} to {ANALYSIS_END}.")

    ds = xr.concat(
        pieces,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        combine_attrs="override",
    ).sortby("time")
    print(f"     synchronous stencil dims: {dict(ds.sizes)}")
    print(f"     loaded stencil lat values: {ds['lat'].values}")
    print(f"     loaded stencil lon values: {ds['lon'].values}")
    return ds


def pressure_values_pa(coord):
    values = np.asarray(coord.values, dtype=float)
    finite = values[np.isfinite(values)]
    units = str(coord.attrs.get("units", "")).lower()
    if any(unit in units for unit in ("hpa", "millibar", "mbar")):
        values = values * 100.0
    elif finite.size and np.nanmax(np.abs(finite)) < 2000.0:
        values = values * 100.0
    return values


def pressure_dataarray_pa(da):
    units = str(da.attrs.get("units", "")).lower()
    finite_max = float(np.nanmax(np.abs(da.values))) if da.size else np.nan
    if any(unit in units for unit in ("hpa", "millibar", "mbar")):
        return da * 100.0
    if np.isfinite(finite_max) and finite_max < 2000.0:
        return da * 100.0
    return da


def build_pressure_thickness_pa(ds3d, ps_2d):
    lev_dim = mb.get_lev_dim(ds3d)

    for key in ("DELP", "delp", "Dp", "dP"):
        if key in ds3d:
            dp = pressure_dataarray_pa(ds3d[key])
            if lev_dim not in dp.dims:
                raise ValueError(f"{key} present but missing '{lev_dim}' dim.")
            return dp.clip(min=0), key

    for key in ("lev_bnds", "plev_bnds", "p_bnds", "pbnds"):
        if key in ds3d:
            pb = pressure_dataarray_pa(ds3d[key])
            bdim = "bnds" if "bnds" in pb.dims else ("bound" if "bound" in pb.dims else "nbnds")
            return abs(pb.isel({bdim: 1}) - pb.isel({bdim: 0})), key

    p_mid = pressure_values_pa(ds3d[lev_dim])
    nlev = int(len(p_mid))
    if nlev < 2:
        raise ValueError("Need at least two pressure levels to infer layer thickness.")

    order = np.argsort(p_mid)
    p_sorted = p_mid[order]
    edges = np.empty(nlev + 1, dtype=float)
    edges[1:-1] = 0.5 * (p_sorted[:-1] + p_sorted[1:])
    edges[0] = max(0.0, p_sorted[0] - 0.5 * (p_sorted[1] - p_sorted[0]))

    pieces = [None] * nlev
    for sorted_idx, orig_idx in enumerate(order):
        upper = edges[sorted_idx]
        if sorted_idx == nlev - 1:
            lower = ps_2d
        else:
            lower = xr.where(ps_2d < edges[sorted_idx + 1], ps_2d, edges[sorted_idx + 1])
        dp = (lower - upper).clip(min=0)
        pieces[int(orig_idx)] = dp.expand_dims({lev_dim: [ds3d[lev_dim].values[int(orig_idx)]]})

    return xr.concat(pieces, dim=lev_dim).assign_coords({lev_dim: ds3d[lev_dim]}), "pressure_levels"


def pressure_midpoint_pa(ds, lev_dim, template_name="QV"):
    return xr.DataArray(pressure_values_pa(ds[lev_dim]), dims=(lev_dim,)).broadcast_like(ds[template_name])


def derivative_radians(da, dim):
    original_coord = da[dim]
    rad_coord = xr.DataArray(np.deg2rad(original_coord.values.astype(float)), dims=(dim,))
    edge_order = 2 if da.sizes.get(dim, 0) >= 3 else 1
    deriv = da.assign_coords({dim: rad_coord}).differentiate(dim, edge_order=edge_order)
    return deriv.assign_coords({dim: original_coord})


def horizontal_divergence_sphere(flux_u, flux_v):
    if flux_u.sizes.get("lon", 0) < 3 or flux_u.sizes.get("lat", 0) < 3:
        raise ValueError("Direct divergence needs at least 3 lat and 3 lon points in the loaded stencil.")

    dfluxu_dlon = derivative_radians(flux_u, "lon")
    dfluxv_dlat = derivative_radians(flux_v, "lat")
    coslat = xr.DataArray(np.cos(np.deg2rad(flux_u["lat"].values.astype(float))), dims=("lat",), coords={"lat": flux_u["lat"]})
    return dfluxu_dlon / (EARTH_RADIUS_M * coslat) + dfluxv_dlat / EARTH_RADIUS_M


def compute_direct_mc_series(prog_patterns, surf_patterns, name, region):
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{name}_{region['key']}_direct_mc.nc"
    if cache_file.exists():
        print(f"\nLoading {name} {region['title']} from cache: {cache_file}")
        ds = xr.open_dataset(cache_file).load()
        print(f"  cached variables: {list(ds.data_vars)}")
        for var in ds.data_vars:
            print(f"    {var}: dims={ds[var].dims}, units={ds[var].attrs.get('units', 'unknown')}")
        return ds

    describe_source_files(prog_patterns, PROG_KEEP, f"{name} prog")
    describe_source_files(surf_patterns, SURF_KEEP, f"{name} surf")

    lat0, lon0 = point_from_region(region)
    print(
        f"\nLoading {name} data for {region['title']} synchronously around "
        f"lat={lat0:g}, lon={lon0:g} with {POINT_STENCIL_COUNT}x{POINT_STENCIL_COUNT} stencil..."
    )
    state3d = load_sync_stencil_patterns(prog_patterns, preprocess_prog(region), f"{name} prog wtp_direct")
    flux2d = load_sync_stencil_patterns(surf_patterns, preprocess_surf(region), f"{name} surf wtp_direct")
    state3d, flux2d, align_freq = mb.align_time_axes(state3d, flux2d, f"{name} {region['title']} direct MC")
    ds = xr.merge([state3d, flux2d], join="inner", compat="override")
    ds = ds.sel(time=slice(ANALYSIS_START, ANALYSIS_END))

    u_name = first_existing(ds, U_CANDIDATES, "zonal wind")
    v_name = first_existing(ds, V_CANDIDATES, "meridional wind")
    precip_name = next((n for n in ("PRECTOT", "TPREC", "precip") if n in ds), None)
    print(f"\n{name} loaded diagnostic variables")
    print(f"  alignment: {align_freq}, time steps: {ds.sizes.get('time', 0)}")
    print(f"  q: QV, u: {u_name}, v: {v_name}, ps: PS, evap flux: LHFX, precip: {precip_name or 'missing'}")
    print(f"  dimensions: {dict(ds.sizes)}")

    lev_dim = mb.get_lev_dim(ds)
    dp, dp_source = build_pressure_thickness_pa(ds, ds["PS"])
    p_mid_4d = pressure_midpoint_pa(ds, lev_dim)
    below_ground = p_mid_4d > ds["PS"]
    q = ds["QV"].where(~below_ground)
    u = ds[u_name].where(~below_ground)
    v = ds[v_name].where(~below_ground)

    col_W_grid = (q * dp / mb.g).sum(lev_dim, skipna=True)
    flux_u_grid = (q * u * dp / mb.g).sum(lev_dim, skipna=True)
    flux_v_grid = (q * v * dp / mb.g).sum(lev_dim, skipna=True)
    mc_direct_grid = -horizontal_divergence_sphere(flux_u_grid, flux_v_grid)

    target_template = select_nearest_point(col_W_grid, region)
    target_lats = target_template["lat"].values
    target_lons = target_template["lon"].values

    col_W = mb.box_mean(col_W_grid.sel(lat=target_lats, lon=target_lons))
    Precip = (
        mb.box_mean(ds[precip_name].sel(lat=target_lats, lon=target_lons))
        if precip_name
        else xr.zeros_like(col_W)
    )
    Evap = mb.box_mean((mb.ensure_atm_positive(ds["LHFX"]) / mb.Lv).sel(lat=target_lats, lon=target_lons))
    MCDirect = mb.box_mean(mc_direct_grid.sel(lat=target_lats, lon=target_lons))
    StorageRelease = -col_W.differentiate("time", datetime_unit="s")
    MCResidual = Precip - Evap - StorageRelease
    ClosureResidual = Precip - Evap - MCDirect - StorageRelease

    out = xr.Dataset(
        {
            "Precip": Precip,
            "Evap": Evap,
            "col_W": col_W,
            "StorageRelease": StorageRelease,
            "MCDirect": MCDirect,
            "MCResidual": MCResidual,
            "ClosureResidual": ClosureResidual,
        },
        attrs={
            "experiment_name": name,
            "region_name": region["title"],
            "region_key": region["key"],
            "align_freq": align_freq,
            "u_variable": u_name,
            "v_variable": v_name,
            "precip_variable": precip_name or "",
            "pressure_thickness_source": dp_source,
            "target_lat_values": ",".join(str(float(x)) for x in np.atleast_1d(target_lats)),
            "target_lon_values": ",".join(str(float(x)) for x in np.atleast_1d(target_lons)),
            "point_stencil_count": POINT_STENCIL_COUNT,
        },
    )
    for var in ("Precip", "Evap", "StorageRelease", "MCDirect", "MCResidual", "ClosureResidual"):
        out[var].attrs["units"] = "kg m-2 s-1 (equivalent mm s-1)"
    out["col_W"].attrs["units"] = "kg m-2 (equivalent mm)"

    print(f"  pressure thickness source: {dp_source}")
    print(f"  target lat values: {target_lats}")
    print(f"  target lon values: {target_lons}")
    print(f"  column W range: {float(col_W.min()):.2f} to {float(col_W.max()):.2f} mm")
    print(f"  direct MC range: {float(MCDirect.min() * SECONDS_PER_DAY):.2f} to {float(MCDirect.max() * SECONDS_PER_DAY):.2f} mm day-1")
    print(f"  residual MC range: {float(MCResidual.min() * SECONDS_PER_DAY):.2f} to {float(MCResidual.max() * SECONDS_PER_DAY):.2f} mm day-1")

    tmp_cache_file = cache_file.with_suffix(cache_file.suffix + ".tmp")
    tmp_cache_file.unlink(missing_ok=True)
    out.to_netcdf(tmp_cache_file)
    tmp_cache_file.replace(cache_file)
    print(f"  cached direct-MC series to {cache_file}")
    return out


def integrate_flux_window(da):
    return mb.integrate_flux_window(da)


def build_event_series(ds, smooth_hours=0.0):
    steps = mb.smooth_steps_from_hours(ds, smooth_hours)
    out = xr.Dataset(attrs=dict(ds.attrs))
    for name in ("Precip", "Evap", "MCDirect", "MCResidual", "ClosureResidual"):
        out[name] = mb.rolling_event_mean(ds[name], steps)
    out["col_W"] = mb.rolling_event_mean(ds["col_W"], steps)
    out["StorageRelease"] = -out["col_W"].differentiate("time", datetime_unit="s")
    return out


def build_event_table(series, spike_indices, label, window_hours=EVENT_WINDOW_HOURS):
    half_steps = mb.half_window_steps_from_hours(series, window_hours)
    rows = []
    for spike_id, idx in enumerate(spike_indices, start=1):
        i0 = max(idx - half_steps, 0)
        i1 = min(idx + half_steps, series.sizes["time"] - 1)
        window = series.isel(time=slice(i0, i1 + 1))
        precip_int = integrate_flux_window(window["Precip"])
        evap_int = integrate_flux_window(window["Evap"])
        mc_direct_int = integrate_flux_window(window["MCDirect"])
        storage_int = -(float(series["col_W"].isel(time=i1)) - float(series["col_W"].isel(time=i0)))
        mc_residual_int = precip_int - evap_int - storage_int
        closure_residual_int = precip_int - evap_int - mc_direct_int - storage_int
        rows.append(
            {
                "experiment": label,
                "event_id": f"S{spike_id}",
                "peak_time": np.datetime_as_string(window.time.isel(time=idx - i0).values, unit="h"),
                "event_start": np.datetime_as_string(window.time.isel(time=0).values, unit="h"),
                "event_end": np.datetime_as_string(window.time.isel(time=-1).values, unit="h"),
                "precip_mm": precip_int,
                "evap_mm": evap_int,
                "mc_direct_mm": mc_direct_int,
                "storage_mm": storage_int,
                "mc_residual_mm": mc_residual_int,
                "closure_residual_mm": closure_residual_int,
            }
        )
    return rows, half_steps


def summarize_rows(rows, label):
    print(f"\n{label} WTP direct-MC event summary")
    keys = ("precip_mm", "evap_mm", "mc_direct_mm", "storage_mm", "mc_residual_mm", "closure_residual_mm")
    summary = {}
    for key in keys:
        values = np.array([row[key] for row in rows], dtype=float)
        summary[key] = float(np.nanmean(values))
    print(
        "  mean [mm per 24-h window]: "
        f"P={summary['precip_mm']:.2f}, E={summary['evap_mm']:.2f}, "
        f"MC_direct={summary['mc_direct_mm']:.2f}, -dW={summary['storage_mm']:.2f}, "
        f"MC_residual={summary['mc_residual_mm']:.2f}, "
        f"closure_residual={summary['closure_residual_mm']:.2f}"
    )
    for row in rows:
        print(
            f"  {row['event_id']} {row['peak_time']}: "
            f"P={row['precip_mm']:.2f}, E={row['evap_mm']:.2f}, "
            f"MC_direct={row['mc_direct_mm']:.2f}, -dW={row['storage_mm']:.2f}, "
            f"MC_residual={row['mc_residual_mm']:.2f}, "
            f"closure_residual={row['closure_residual_mm']:.2f}"
        )
    return summary


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def paired_wilcoxon_pvalue(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return np.nan
    if np.allclose(values, 0.0):
        return 1.0
    nonzero = values[~np.isclose(values, 0.0)]
    if nonzero.size <= 1:
        return 1.0
    return float(
        stats.wilcoxon(
            values,
            zero_method="wilcox",
            alternative="two-sided",
            method="auto",
        ).pvalue
    )


def sample_sem(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0
    return float(np.nanstd(values, ddof=1) / np.sqrt(values.size))


def plot_ylim_from_values(values, errors=None, pad_fraction=0.16):
    values = np.asarray(values, dtype=float)
    finite = [0.0]
    if errors is None:
        errors = np.zeros_like(values)
    errors = np.asarray(errors, dtype=float)
    for value, err in zip(values, errors):
        finite.extend([value - err, value + err])
    finite = np.asarray(finite, dtype=float)
    finite = finite[np.isfinite(finite)]
    ymin = float(np.nanmin(finite))
    ymax = float(np.nanmax(finite))
    span = max(ymax - ymin, 1.0)
    pad = max(0.6, pad_fraction * span)
    return ymin - pad, ymax + pad


def summarize_plot_components(me_rows, rp_rows):
    summary = []
    for key, label in PLOT_COMPONENTS:
        me_vals = np.array([row[key] for row in me_rows], dtype=float)
        rp_vals = np.array([row[key] for row in rp_rows], dtype=float)
        diff_vals = me_vals - rp_vals
        summary.append(
            {
                "component_key": key,
                "component_label": label,
                "n_events": int(diff_vals.size),
                "rean_mean_mm": float(np.nanmean(me_vals)),
                "rean_sem_mm": sample_sem(me_vals),
                "iau_mean_mm": float(np.nanmean(rp_vals)),
                "iau_sem_mm": sample_sem(rp_vals),
                "diff_mean_mm": float(np.nanmean(diff_vals)),
                "diff_sem_mm": sample_sem(diff_vals),
                "pvalue": paired_wilcoxon_pvalue(diff_vals),
            }
        )
    return summary


def plot_mean_budget(me_rows, rp_rows):
    summary = summarize_plot_components(me_rows, rp_rows)
    labels = [row["component_label"] for row in summary]
    x = np.arange(len(labels))
    width = 0.42
    pair_offset = 0.24

    rean_mean = np.array([row["rean_mean_mm"] for row in summary], dtype=float)
    iau_mean = np.array([row["iau_mean_mm"] for row in summary], dtype=float)
    rean_sem = np.array([row["rean_sem_mm"] for row in summary], dtype=float)
    iau_sem = np.array([row["iau_sem_mm"] for row in summary], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(11, 6.5))
    fig.subplots_adjust(wspace=0.25)

    ax = axes[0]
    ax.bar(
        x - pair_offset,
        rean_mean,
        width=width,
        color="navy",
        alpha=0.9,
        yerr=rean_sem,
        capsize=4,
        label="Dynamically Imbalanced",
    )
    ax.bar(
        x + pair_offset,
        iau_mean,
        width=width,
        color="darkorange",
        alpha=0.9,
        yerr=iau_sem,
        capsize=4,
        label="Dynamically Balanced",
    )
    ax.axhline(0.0, color="0.4", linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"mm per {EVENT_WINDOW_HOURS:g}-h spike window")
    ax.set_title("(a) WTP mean spike-window moisture budget", loc="left", fontweight="bold", fontsize=12)
    ax.legend(loc="upper left", frameon=False)
    ax.set_ylim(*plot_ylim_from_values(np.r_[rean_mean, iau_mean], np.r_[rean_sem, iau_sem]))

    ax = axes[1]
    diff_mean = np.array([row["diff_mean_mm"] for row in summary], dtype=float)
    diff_sem = np.array([row["diff_sem_mm"] for row in summary], dtype=float)
    colors = ["black", "tab:blue", "tab:green", "tab:purple"]
    bars = ax.bar(x, diff_mean, width=width, color=colors, alpha=0.88, yerr=diff_sem, capsize=4)
    bars[-1].set_hatch("//")
    bars[-1].set_edgecolor("firebrick")
    bars[-1].set_linewidth(1.3)
    ax.axhline(0.0, color="0.4", linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"Difference [mm per {EVENT_WINDOW_HOURS:g}-h spike window]")
    ax.set_title("(b) Mean difference (imbalanced - balanced)", loc="left", fontweight="bold", fontsize=12)

    ymin, ymax = plot_ylim_from_values(diff_mean, diff_sem, pad_fraction=0.22)
    ax.set_ylim(ymin, ymax)
    inset = max(0.35, 0.04 * (ymax - ymin))
    label_pad = max(0.35, 0.04 * (ymax - ymin))
    for xpos, row, value, err in zip(x, summary, diff_mean, diff_sem):
        yloc = value + err + label_pad if value >= 0.0 else value - err - label_pad
        yloc = float(np.clip(yloc, ymin + inset, ymax - inset))
        ax.text(xpos, yloc, f"p={row['pvalue']:.3f}", ha="center", va="center", fontsize=9, clip_on=True)

    for ax in axes:
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.tick_params(axis="both", labelsize=10)

    fig.tight_layout()
    plt.savefig(MEAN_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Mean WTP direct-MC budget figure saved to {MEAN_FIGURE_PATH}")


def annotate_bar_values(ax, bars, values):
    ymin, ymax = ax.get_ylim()
    pad = 0.025 * max(ymax - ymin, 1.0)
    for bar, value in zip(bars, values):
        if value >= 0.0:
            yloc = value + pad
            va = "bottom"
        else:
            yloc = value - pad
            va = "top"
        ax.text(
            bar.get_x() + bar.get_width() * 0.5,
            yloc,
            f"{value:.1f}",
            ha="center",
            va=va,
            fontsize=9,
            clip_on=True,
        )


def plot_story_budget(me_detect, rp_detect, spike_indices, me_rows, rp_rows):
    if len(spike_indices) == 0:
        print("Skipping story figure because no WTP spikes were detected.")
        return

    spike_values = np.asarray(me_detect["Precip"].isel(time=spike_indices).values, dtype=float)
    dominant_position = int(np.nanargmax(spike_values))
    dominant_idx = int(spike_indices[dominant_position])
    me_event = me_rows[dominant_position]
    rp_event = rp_rows[dominant_position]
    diff_event = {key: me_event[key] - rp_event[key] for key, _ in PLOT_COMPONENTS}

    time_values = me_detect.time.values
    half_steps = mb.half_window_steps_from_hours(me_detect, EVENT_WINDOW_HOURS)
    event_start_idx = max(dominant_idx - half_steps, 0)
    event_end_idx = min(dominant_idx + half_steps, len(time_values) - 1)
    event_start = time_values[event_start_idx]
    event_end = time_values[event_end_idx]

    dt_hours = mb.infer_time_step_hours(me_detect)
    context_steps = max(1, int(round(12.0 / max(dt_hours, 1.0e-6))))
    plot_start = time_values[max(event_start_idx - context_steps, 0)]
    plot_end = time_values[min(event_end_idx + context_steps, len(time_values) - 1)]

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 11.0), sharex=False)
    fig.subplots_adjust(hspace=0.30, top=0.93)
    fig.suptitle(
        "WTP direct-MC moisture budget of the dominant precipitation spike",
        fontsize=15,
        y=0.985,
    )

    ax = axes[0]
    ax.axvspan(event_start, event_end, color="0.86", alpha=0.55, linewidth=0)
    me_focus = me_detect.sel(time=slice(plot_start, plot_end))
    rp_focus = rp_detect.sel(time=slice(plot_start, plot_end))
    ax.plot(
        me_focus.time.values,
        me_focus["Precip"] * SECONDS_PER_DAY,
        color="navy",
        linewidth=2.8,
        label="Dynamically Imbalanced",
    )
    ax.plot(
        rp_focus.time.values,
        rp_focus["Precip"] * SECONDS_PER_DAY,
        color="darkorange",
        linewidth=2.3,
        label="Dynamically Balanced",
    )
    ax.scatter(
        [time_values[dominant_idx]],
        [float(me_detect["Precip"].isel(time=dominant_idx) * SECONDS_PER_DAY)],
        color="navy",
        s=34,
        zorder=5,
    )
    ax.annotate(
        me_event["event_id"],
        (time_values[dominant_idx], float(me_detect["Precip"].isel(time=dominant_idx) * SECONDS_PER_DAY)),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=10,
    )
    ax.set_xlim(plot_start, plot_end)
    ax.set_ylabel(r"$P$ [mm day$^{-1}$]")
    ax.set_title("(a) Precipitation time series", loc="left", fontweight="bold")
    ax.legend(loc="upper right", ncol=2, frameon=False, fontsize=10)

    labels = [label for _, label in PLOT_COMPONENTS]
    x = np.arange(len(labels))
    colors = ["black", "tab:blue", "tab:green", "tab:purple"]

    ax = axes[1]
    me_values = np.array([me_event[key] for key, _ in PLOT_COMPONENTS], dtype=float)
    bars = ax.bar(x, me_values, color=colors, alpha=0.88, width=0.72, edgecolor="none")
    bars[-1].set_hatch("//")
    bars[-1].set_edgecolor("firebrick")
    bars[-1].set_linewidth(1.3)
    ax.axhline(0.0, color="0.4", linewidth=1.0, linestyle="--")
    ax.set_ylabel(f"mm per {EVENT_WINDOW_HOURS:g}-h spike window")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_title("(b) Event-integrated WTP direct-MC budget", loc="left", fontweight="bold")
    ax.set_ylim(*plot_ylim_from_values(me_values))
    annotate_bar_values(ax, bars, me_values)

    ax = axes[2]
    diff_values = np.array([diff_event[key] for key, _ in PLOT_COMPONENTS], dtype=float)
    diff_bars = ax.bar(x, diff_values, color=colors, alpha=0.88, width=0.72, edgecolor="none")
    diff_bars[-1].set_hatch("//")
    diff_bars[-1].set_edgecolor("firebrick")
    diff_bars[-1].set_linewidth(1.3)
    ax.axhline(0.0, color="0.4", linewidth=1.0, linestyle="--")
    ax.set_ylabel("Imbalanced - balanced\n[mm]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_title("(c) Event-integrated difference", loc="left", fontweight="bold")
    ax.set_ylim(*plot_ylim_from_values(diff_values))
    annotate_bar_values(ax, diff_bars, diff_values)

    for ax in axes:
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.tick_params(axis="both", labelsize=10)

    axes[0].xaxis.set_major_locator(mdates.HourLocator(interval=6))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%HZ"))
    axes[0].set_xlabel("Time around selected WTP spike")
    axes[-1].set_xlabel("Integrated budget terms")

    fig.tight_layout()
    plt.savefig(STORY_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Story WTP direct-MC budget figure saved to {STORY_FIGURE_PATH}")


def main():
    me_prog_patterns = monthly_patterns(mb.me_prog)
    me_surf_patterns = monthly_patterns(mb.me_surf)
    rp_prog_patterns = monthly_patterns(mb.rp_prog)
    rp_surf_patterns = monthly_patterns(mb.rp_surf)

    me_series = compute_direct_mc_series(me_prog_patterns, me_surf_patterns, "Reanalysis-IC", WTP_TARGET)
    rp_series = compute_direct_mc_series(rp_prog_patterns, rp_surf_patterns, "IAU-IC", WTP_TARGET)
    me_series, rp_series = xr.align(me_series, rp_series, join="inner")

    me_detect = build_event_series(me_series, smooth_hours=PRECIP_SMOOTH_HOURS)
    rp_detect = build_event_series(rp_series, smooth_hours=PRECIP_SMOOTH_HOURS)
    reference_precip = xr.where(me_detect["Precip"] >= rp_detect["Precip"], me_detect["Precip"], rp_detect["Precip"])
    dt_hours = mb.infer_time_step_hours(me_detect)
    min_sep_steps = max(1, int(round(MIN_PEAK_SEPARATION_HOURS / max(dt_hours, 1.0e-6))))
    spike_idx, threshold = mb.detect_precip_spikes(
        reference_precip,
        quantile=SPIKE_QUANTILE,
        min_separation=min_sep_steps,
        max_spikes=999,
    )
    print(
        f"\nDetected {len(spike_idx)} WTP spikes above {SPIKE_QUANTILE:.2f} quantile "
        f"(threshold {threshold * SECONDS_PER_DAY:.2f} mm day-1)."
    )

    me_budget = build_event_series(me_series, smooth_hours=0.0)
    rp_budget = build_event_series(rp_series, smooth_hours=0.0)
    me_rows, _ = build_event_table(me_budget, spike_idx, "Reanalysis-IC")
    rp_rows, _ = build_event_table(rp_budget, spike_idx, "IAU-IC")
    rows = me_rows + rp_rows

    fieldnames = [
        "experiment",
        "event_id",
        "peak_time",
        "event_start",
        "event_end",
        "precip_mm",
        "evap_mm",
        "mc_direct_mm",
        "storage_mm",
        "mc_residual_mm",
        "closure_residual_mm",
    ]
    write_csv(EVENT_TABLE_PATH, rows, fieldnames)
    print(f"\nEvent table saved to {EVENT_TABLE_PATH}")

    me_summary = summarize_rows(me_rows, "Reanalysis-IC")
    rp_summary = summarize_rows(rp_rows, "IAU-IC")
    diff_summary = {key: me_summary[key] - rp_summary[key] for key in me_summary}
    print("\nReanalysis-IC minus IAU-IC mean difference")
    print(
        "  "
        f"P={diff_summary['precip_mm']:.2f}, E={diff_summary['evap_mm']:.2f}, "
        f"MC_direct={diff_summary['mc_direct_mm']:.2f}, -dW={diff_summary['storage_mm']:.2f}, "
        f"MC_residual={diff_summary['mc_residual_mm']:.2f}, "
        f"closure_residual={diff_summary['closure_residual_mm']:.2f}"
    )

    summary_rows = [
        {"experiment": "Reanalysis-IC", **me_summary},
        {"experiment": "IAU-IC", **rp_summary},
        {"experiment": "Reanalysis-IC minus IAU-IC", **diff_summary},
    ]
    write_csv(SUMMARY_PATH, summary_rows, ["experiment", *me_summary.keys()])
    print(f"Summary saved to {SUMMARY_PATH}")
    plot_mean_budget(me_rows, rp_rows)
    plot_story_budget(me_detect, rp_detect, spike_idx, me_rows, rp_rows)


if __name__ == "__main__":
    main()
