"""Direct moisture-convergence budget for the Western Tropical Pacific, 2-week window.

Computes vertically integrated horizontal moisture-flux convergence from q, u,
and v on a 3×3 lat/lon stencil centred on the WTP point (143°E, 1°S),
for the first two weeks of the May 2005 forecasts (ME0506 and RP05062).

Cached output is consumed by wp_diag_long.py to overlay MC on the diagnostic
figure.

Based on mse_budget2.py (direct MC calculation).
"""

import glob
from pathlib import Path

import numpy as np
import xarray as xr

import mse_budget as mb


# ==============================================================================
# Configuration — copied from mse_budget2.py, narrowed to WTP + 2 weeks
# ==============================================================================
ANALYSIS_START = "2005-05-06"
ANALYSIS_END = "2005-05-20"
SECONDS_PER_DAY = 86400.0
EARTH_RADIUS_M = 6_371_000.0
POINT_STENCIL_COUNT = 3

# Western Tropical Pacific point (same as mse_budget2.py)
WTP_REGION = {
    "key": "wtp",
    "title": "Western Tropical Pacific",
    "lon": (143.0, 143.0),
    "lat": (-1.0, -1.0),
}

# Experiment paths from mse_budget.py
ME_PROG = mb.me_prog
ME_SURF = mb.me_surf
RP_PROG = mb.rp_prog
RP_SURF = mb.rp_surf

# Only need May (the 2-week window is entirely within 200505)
MONTHS = ("200505",)

# Variables to keep when loading
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

# Cache
CACHE_DIR = Path(__file__).with_name("cache_mse_budget_wp_2wk")


# ==============================================================================
# File helpers
# ==============================================================================
def monthly_patterns(pattern, months=MONTHS):
    """Expand a glob pattern across the requested months."""
    return [pattern.replace("200505", month) for month in months]


def expand_files(patterns):
    """Expand a list of glob patterns into a sorted, deduplicated file list."""
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


# ==============================================================================
# Variable / coordinate helpers
# ==============================================================================
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


def stencil_coord_values(ds, region, count=POINT_STENCIL_COUNT):
    lat0, lon0 = point_from_region(region)
    lat_vals = mb.nearest_coord_values(ds["lat"], lat0, count, is_lon=False)
    lon_target = mb.canon_lon(lon0, ds["lon"])
    lon_vals = mb.nearest_coord_values(ds["lon"], lon_target, count, is_lon=True)
    return lat_vals, lon_vals


def select_nearest_stencil(ds, region, count=POINT_STENCIL_COUNT):
    lat_vals, lon_vals = stencil_coord_values(ds, region, count=count)
    return ds.sel(lat=lat_vals, lon=lon_vals)


def select_nearest_point(ds, region):
    lat0, lon0 = point_from_region(region)
    lat_vals = mb.nearest_coord_values(ds["lat"], lat0, 1, is_lon=False)
    lon_target = mb.canon_lon(lon0, ds["lon"])
    lon_vals = mb.nearest_coord_values(ds["lon"], lon_target, 1, is_lon=True)
    return ds.sel(lat=lat_vals, lon=lon_vals)


# ==============================================================================
# Pressure helpers (from mse_budget2.py)
# ==============================================================================
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


# ==============================================================================
# Direct MC calculation (from mse_budget2.py)
# ==============================================================================
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
    coslat = xr.DataArray(
        np.cos(np.deg2rad(flux_u["lat"].values.astype(float))),
        dims=("lat",),
        coords={"lat": flux_u["lat"]},
    )
    return dfluxu_dlon / (EARTH_RADIUS_M * coslat) + dfluxv_dlat / EARTH_RADIUS_M


# ==============================================================================
# Preprocessing for lazy/serial loading
# ==============================================================================
def preprocess_prog(ds):
    ds = subset_existing(ds, PROG_KEEP, ("QV",), "prog")
    first_existing(ds, U_CANDIDATES, "zonal wind")
    first_existing(ds, V_CANDIDATES, "meridional wind")
    return select_nearest_stencil(ds, WTP_REGION)


def preprocess_surf(ds):
    ds = subset_existing(ds, SURF_KEEP, ("PS", "LHFX"), "surf")
    return select_nearest_stencil(ds, WTP_REGION)


# ==============================================================================
# Data loading
# ==============================================================================
def load_serial_patterns(patterns, preprocess, label):
    """Load files one-by-one, applying preprocessing and time subsetting."""
    files = expand_files(patterns)
    if not files:
        raise ValueError(f"No files matched for {label}: {patterns}")

    print(f"  -> Opening {len(files)} {label} files synchronously with WTP stencil.")
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
    print(f"     stencil dims: {dict(ds.sizes)}")
    print(f"     loaded lat values: {ds['lat'].values}")
    print(f"     loaded lon values: {ds['lon'].values}")
    return ds


# ==============================================================================
# Direct MC budget computation
# ==============================================================================
def build_direct_mc_series(ds, name, align_freq):
    """Compute the direct-MC moisture budget from a merged prog+surf dataset."""
    ds = select_nearest_stencil(ds, WTP_REGION)

    u_name = first_existing(ds, U_CANDIDATES, "zonal wind")
    v_name = first_existing(ds, V_CANDIDATES, "meridional wind")
    precip_name = next((n for n in ("PRECTOT", "TPREC", "precip") if n in ds), None)

    print(f"\n{name} loaded diagnostic variables for {WTP_REGION['title']}")
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

    # Column-integrated moisture and fluxes
    col_W_grid = (q * dp / mb.g).sum(lev_dim, skipna=True)
    flux_u_grid = (q * u * dp / mb.g).sum(lev_dim, skipna=True)
    flux_v_grid = (q * v * dp / mb.g).sum(lev_dim, skipna=True)
    mc_direct_grid = -horizontal_divergence_sphere(flux_u_grid, flux_v_grid)

    # Extract nearest point from the stencil
    target_template = select_nearest_point(col_W_grid, WTP_REGION)
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
            "region_name": WTP_REGION["title"],
            "region_key": WTP_REGION["key"],
            "align_freq": align_freq,
            "u_variable": u_name,
            "v_variable": v_name,
            "precip_variable": precip_name or "",
            "pressure_thickness_source": dp_source,
            "target_lat_values": ",".join(str(float(x)) for x in np.atleast_1d(target_lats)),
            "target_lon_values": ",".join(str(float(x)) for x in np.atleast_1d(target_lons)),
            "point_stencil_count": POINT_STENCIL_COUNT,
            "analysis_start": ANALYSIS_START,
            "analysis_end": ANALYSIS_END,
        },
    )
    for var in ("Precip", "Evap", "StorageRelease", "MCDirect", "MCResidual", "ClosureResidual"):
        out[var].attrs["units"] = "kg m-2 s-1 (equivalent mm s-1)"
    out["col_W"].attrs["units"] = "kg m-2 (equivalent mm)"

    print(f"  pressure thickness source: {dp_source}")
    print(f"  target lat values: {target_lats}")
    print(f"  target lon values: {target_lons}")
    print(f"  column W range: {float(col_W.min()):.2f} to {float(col_W.max()):.2f} mm")
    print(f"  MC range: {float(MCDirect.min() * SECONDS_PER_DAY):.2f} to {float(MCDirect.max() * SECONDS_PER_DAY):.2f} mm day-1")
    print(f"  residual MC range: {float(MCResidual.min() * SECONDS_PER_DAY):.2f} to {float(MCResidual.max() * SECONDS_PER_DAY):.2f} mm day-1")
    return out


def compute_experiment(prog_pattern, surf_pattern, name):
    """Load, merge, and compute the direct-MC budget for one experiment."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{name}_wtp_direct_mc.nc"

    if cache_file.exists():
        print(f"\nLoading {name} WTP from cache: {cache_file}")
        ds = xr.open_dataset(cache_file).load()
        print(f"  cached variables: {list(ds.data_vars)}")
        return ds

    prog_patterns = monthly_patterns(prog_pattern)
    surf_patterns = monthly_patterns(surf_pattern)

    lat0, lon0 = point_from_region(WTP_REGION)
    print(
        f"\nLoading {name} data for {WTP_REGION['title']} around "
        f"lat={lat0:g}, lon={lon0:g} with {POINT_STENCIL_COUNT}x{POINT_STENCIL_COUNT} stencil..."
    )

    state3d = load_serial_patterns(prog_patterns, preprocess_prog, f"{name} prog")
    flux2d = load_serial_patterns(surf_patterns, preprocess_surf, f"{name} surf")
    state3d, flux2d, align_freq = mb.align_time_axes(state3d, flux2d, f"{name} WTP MC")
    ds = xr.merge([state3d, flux2d], join="inner", compat="override")
    ds = ds.sel(time=slice(ANALYSIS_START, ANALYSIS_END))

    out = build_direct_mc_series(ds, name, align_freq)

    # Write cache
    tmp = cache_file.with_suffix(cache_file.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    out.to_netcdf(tmp)
    tmp.replace(cache_file)
    print(f"  cached to {cache_file}")
    return out


# ==============================================================================
# Main
# ==============================================================================
def main():
    print("=" * 72)
    print("Direct-MC moisture budget: Western Tropical Pacific, 2-week window")
    print(f"  Analysis window : {ANALYSIS_START} to {ANALYSIS_END}")
    print(f"  WTP point       : lat={WTP_REGION['lat']}, lon={WTP_REGION['lon']}")
    print(f"  Stencil          : {POINT_STENCIL_COUNT}x{POINT_STENCIL_COUNT}")
    print(f"  Cache directory  : {CACHE_DIR}")
    print("=" * 72)

    me_series = compute_experiment(ME_PROG, ME_SURF, "Reanalysis-IC")
    rp_series = compute_experiment(RP_PROG, RP_SURF, "IAU-IC")

    # Quick summary
    for label, ds in [("Reanalysis-IC (imbalanced)", me_series), ("IAU-IC (balanced)", rp_series)]:
        print(f"\n{label} summary over {ANALYSIS_START} to {ANALYSIS_END}:")
        print(f"  mean P   = {float(ds['Precip'].mean()) * SECONDS_PER_DAY:7.2f} mm day-1")
        print(f"  mean E   = {float(ds['Evap'].mean()) * SECONDS_PER_DAY:7.2f} mm day-1")
        print(f"  mean MC  = {float(ds['MCDirect'].mean()) * SECONDS_PER_DAY:7.2f} mm day-1")
        print(f"  mean -dW = {float(ds['StorageRelease'].mean()) * SECONDS_PER_DAY:7.2f} mm day-1")
        print(f"  closure  = {float(ds['ClosureResidual'].mean()) * SECONDS_PER_DAY:7.4f} mm day-1")

    print(f"\nCache files written to {CACHE_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
