"""Western Tropical Pacific moisture-budget test with direct moisture convergence.

This diagnostic is intentionally separate from spike_budget_stats.py. It loads a
small halo around the WTP point, computes vertically integrated horizontal
moisture-flux convergence from q, u, and v, then samples the nearest WTP grid
cell.
"""

import csv
import glob
from pathlib import Path

import numpy as np
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
HALO_DEG = 5.0

CACHE_DIR = Path(__file__).with_name("cache_mse_budget2_direct_mc_wtp_point")
EVENT_TABLE_PATH = Path(__file__).with_name("mse_budget2_wtp_direct_mc_events.csv")
SUMMARY_PATH = Path(__file__).with_name("mse_budget2_wtp_direct_mc_summary.csv")

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


def halo_region(region, halo_deg=HALO_DEG):
    return (
        (region["lat"][0] - halo_deg, region["lat"][1] + halo_deg),
        (region["lon"][0] - halo_deg, region["lon"][1] + halo_deg),
    )


def sel_region_with_fallback(ds, lat_rng, lon_rng):
    subset = mb.sel_region(ds, lat_rng, lon_rng)
    if subset.sizes.get("lat", 0) > 0 and subset.sizes.get("lon", 0) > 0:
        return subset

    lat_center = 0.5 * (lat_rng[0] + lat_rng[1])
    lon_center = 0.5 * (lon_rng[0] + lon_rng[1])
    lat_vals = mb.nearest_coord_values(ds["lat"], lat_center, 1, is_lon=False)
    lon_target = mb.canon_lon(lon_center, ds["lon"])
    lon_vals = mb.nearest_coord_values(ds["lon"], lon_target, 1, is_lon=True)
    return ds.sel(lat=lat_vals, lon=lon_vals)


def preprocess_prog(lat_rng, lon_rng):
    def _preprocess(ds):
        ds = subset_existing(ds, PROG_KEEP, ("QV",), "prog")
        first_existing(ds, U_CANDIDATES, "zonal wind")
        first_existing(ds, V_CANDIDATES, "meridional wind")
        return mb.sel_region(ds, lat_rng, lon_rng)

    return _preprocess


def preprocess_surf(lat_rng, lon_rng):
    def _preprocess(ds):
        ds = subset_existing(ds, SURF_KEEP, ("PS", "LHFX"), "surf")
        return mb.sel_region(ds, lat_rng, lon_rng)

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


def load_serial_subset_patterns(patterns, preprocess, label):
    files = expand_files(patterns)
    if not files:
        raise ValueError(f"No files matched for {label}: {patterns}")

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

    return xr.concat(
        pieces,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        combine_attrs="override",
    ).sortby("time")


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
        raise ValueError("Direct divergence needs at least 3 lat and 3 lon points in the loaded halo.")

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

    lat_halo, lon_halo = halo_region(region)
    describe_source_files(prog_patterns, PROG_KEEP, f"{name} prog")
    describe_source_files(surf_patterns, SURF_KEEP, f"{name} surf")

    print(f"\nLoading {name} data for {region['title']} with {HALO_DEG:g} deg halo...")
    state3d = load_serial_subset_patterns(prog_patterns, preprocess_prog(lat_halo, lon_halo), f"{name} prog wtp_direct")
    flux2d = load_serial_subset_patterns(surf_patterns, preprocess_surf(lat_halo, lon_halo), f"{name} surf wtp_direct")
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

    target_lat = region["lat"]
    target_lon = region["lon"]
    target_template = sel_region_with_fallback(col_W_grid, target_lat, target_lon)
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
            "halo_deg": HALO_DEG,
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


if __name__ == "__main__":
    main()
