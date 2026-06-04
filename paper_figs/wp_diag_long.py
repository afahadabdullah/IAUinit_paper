#!/usr/bin/env python3
"""Create the Western Tropical Pacific diagnostic figure.

This script is a standalone version of cape_omg_WP.ipynb. It plots the
dynamically imbalanced and dynamically balanced diagnostics used for
WP_diag_long.png.

Optionally loads direct moisture-convergence cache files produced by
mse_budget_wp_2wk.py and adds MC + Precip panels as a 4th column.
"""

from __future__ import annotations

import argparse
import re
from datetime import date, datetime
from pathlib import Path

import dask
import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import xarray as xr


PRECIP_CANDIDATES = ("PRECTOT", "PRECTOTCORR", "PR", "PRECCON", "PRECLS")
CAPE_VARS = ("CAPE", "T", "OMEGA")
SURF_VARS = ("SHFX", "TA", "TS_FOUND", "PRECTOT", "PRECTOTCORR", "PR", "PRECCON", "PRECLS")


def progress(message: str) -> None:
    print(f"[wp_diag] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Western Tropical Pacific CAPE, omega, flux, SST, and precipitation diagnostics."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/nobackupp27/afahad/exp/IAU_exp"),
        help="Directory containing GEOSMIT experiment folders.",
    )
    parser.add_argument(
        "--imbalanced-exp",
        default="GEOSMIT_ME0506",
        help="Experiment initialized from the dynamically imbalanced state.",
    )
    parser.add_argument(
        "--balanced-exp",
        default="GEOSMIT_RP05062",
        help="Experiment initialized from the dynamically balanced state.",
    )
    parser.add_argument("--month", default="200505", help="YYYYMM holding subdirectory.")
    parser.add_argument("--start", default="2005-05-06", help="Start date.")
    parser.add_argument("--end", default="2005-05-20", help="End date.")
    parser.add_argument("--lon", type=float, default=143.0, help="Longitude for the diagnostic point.")
    parser.add_argument("--lat", type=float, default=-1.0, help="Latitude for the diagnostic point.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("WP_diag_long.png"),
        help="Output figure path.",
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=None,
        help="Processed point time-series cache. Defaults to <output stem>_cache.nc4.",
    )
    parser.add_argument(
        "--recompute-cache",
        action="store_true",
        help="Ignore any existing processed cache and reread the source NetCDF files.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not read or write the processed point time-series cache.",
    )
    parser.add_argument(
        "--time-chunk",
        type=int,
        default=16,
        help="Time chunk size used while lazily reading source NetCDF files.",
    )
    parser.add_argument(
        "--mc-cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing direct-MC cache files from mse_budget_wp_2wk.py. "
            "Defaults to cache_mse_budget_wp_2wk/ next to this script."
        ),
    )
    return parser.parse_args()


def parse_date(value: str) -> date:
    return datetime.fromisoformat(value).date()


def file_date(path: Path) -> date | None:
    match = re.search(r"(\d{8})(?:_\d{4}z)?", path.name)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d").date()


def collection_files(
    base_dir: Path,
    exp: str,
    collection: str,
    month: str,
    start: str,
    end: str,
) -> list[str]:
    pattern = base_dir / exp / "holding" / collection / month / f"*{collection.split('_')[-1]}*{month}*z.nc4"
    start_date = parse_date(start)
    end_date = parse_date(end)
    files = []
    for path in sorted(pattern.parent.glob(pattern.name)):
        date = file_date(path)
        if date is not None and not (start_date <= date <= end_date):
            continue
        files.append(str(path))
    if not files:
        raise FileNotFoundError(f"No files found for {exp} {collection}: {pattern}")
    progress(f"{exp} {collection}: found {len(files)} files in {month} for {start} to {end}")
    return files


def subset_dataset(
    ds: xr.Dataset,
    variables: tuple[str, ...],
    start: str,
    end: str,
    lat: float,
    lon: float,
) -> xr.Dataset:
    present = [name for name in variables if name in ds]
    if not present:
        return ds[[]]

    ds = ds[present]
    if "time" in ds.coords or "time" in ds.dims:
        ds = ds.sel(time=slice(start, end))
    if "lev" in ds.coords:
        ds = ds.sel(lev=500, method="nearest")
    if "lat" in ds.coords and "lon" in ds.coords:
        ds = ds.sel(lat=lat, lon=lon, method="nearest")
    return ds


def open_collection(
    base_dir: Path,
    exp: str,
    collection: str,
    month: str,
    start: str,
    end: str,
    variables: tuple[str, ...],
    lat: float,
    lon: float,
    time_chunk: int,
) -> xr.Dataset:
    files = collection_files(base_dir, exp, collection, month, start, end)
    progress(f"{exp} {collection}: opening lazily with variables {', '.join(variables)}")
    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        preprocess=lambda data: subset_dataset(data, variables, start, end, lat, lon),
        chunks={"time": time_chunk},
        data_vars="minimal",
        coords="minimal",
        compat="override",
    )
    return ds.sel(time=slice(start, end))


def point_series(da: xr.DataArray) -> xr.DataArray:
    dims = [dim for dim in ("lat", "lon") if dim in da.dims]
    if dims:
        da = da.mean(dim=dims)
    return da.squeeze(drop=True)


def precip_from_surf(ds: xr.Dataset) -> xr.DataArray:
    for name in ("PRECTOT", "PRECTOTCORR", "PR"):
        if name in ds:
            precip = ds[name]
            break
    else:
        if "PRECCON" in ds and "PRECLS" in ds:
            precip = ds["PRECCON"] + ds["PRECLS"]
            precip.attrs["units"] = ds["PRECCON"].attrs.get("units", "")
            precip.name = "PRECTOT"
        else:
            available = ", ".join(sorted(ds.data_vars))
            raise KeyError(
                "Could not find precipitation. Tried "
                f"{', '.join(PRECIP_CANDIDATES)}; available variables: {available}"
            )

    units = str(precip.attrs.get("units", "")).lower()
    if "s-1" in units or "/s" in units or "sec" in units:
        precip = precip * 86400.0
        precip.attrs["units"] = "mm day-1"
    elif not units:
        precip = precip * 86400.0
        precip.attrs["units"] = "mm day-1"
    return precip


def load_case(
    base_dir: Path,
    exp: str,
    month: str,
    start: str,
    end: str,
    lat: float,
    lon: float,
    time_chunk: int,
) -> dict[str, xr.DataArray]:
    progress(f"{exp}: loading point diagnostics near lat={lat}, lon={lon}")
    cape_ds = open_collection(
        base_dir, exp, "geosgcm_cape", month, start, end, CAPE_VARS, lat, lon, time_chunk
    )
    surf_ds = open_collection(
        base_dir, exp, "geosgcm_surf", month, start, end, SURF_VARS, lat, lon, time_chunk
    )

    try:
        progress(f"{exp}: computing CAPE, omega, temperature, flux, SST, and precipitation time series")
        cape = point_series(cape_ds["CAPE"]).compute()
        temp = point_series(cape_ds["T"]).compute()
        omega = point_series(cape_ds["OMEGA"]).compute()

        shfx = point_series(surf_ds["SHFX"]).compute()
        air_temp = point_series(surf_ds["TA"]).compute()
        sst = point_series(surf_ds["TS_FOUND"]).compute()
        sst_tendency = sst.diff("time").compute()
        precip = point_series(precip_from_surf(surf_ds)).compute()

        vertical_heat_flux = (-temp * omega)[2::3]
    finally:
        cape_ds.close()
        surf_ds.close()

    progress(f"{exp}: finished processed point time series")

    return {
        "cape": cape,
        "omega": omega,
        "precip": precip,
        "shfx": shfx,
        "vertical_heat_flux": vertical_heat_flux,
        "air_temp": air_temp,
        "sst_tendency": sst_tendency,
    }


def default_cache_file(output: Path) -> Path:
    return output.with_name(f"{output.stem}_cache.nc4")


def cache_attrs(args: argparse.Namespace) -> dict[str, str]:
    return {
        "base_dir": str(args.base_dir),
        "imbalanced_exp": args.imbalanced_exp,
        "balanced_exp": args.balanced_exp,
        "month": args.month,
        "start": args.start,
        "end": args.end,
        "lat": str(args.lat),
        "lon": str(args.lon),
    }


def cache_matches(ds: xr.Dataset, args: argparse.Namespace) -> bool:
    expected = cache_attrs(args)
    return all(str(ds.attrs.get(key)) == value for key, value in expected.items())


def cases_to_dataset(cases: dict[str, dict[str, xr.DataArray]], args: argparse.Namespace) -> xr.Dataset:
    data_vars = {}
    for case_name, case in cases.items():
        for var_name, da in case.items():
            dim_name = f"time__{case_name}__{var_name}"
            cached = da.rename({"time": dim_name}).reset_coords(drop=True)
            data_vars[f"{case_name}__{var_name}"] = cached
    return xr.Dataset(data_vars, attrs=cache_attrs(args))


def cache_encoding(ds: xr.Dataset, args: argparse.Namespace) -> dict[str, dict[str, str]]:
    origin = datetime.fromisoformat(args.start).strftime("%Y-%m-%d %H:%M:%S")
    return {
        coord: {"units": f"hours since {origin}"}
        for coord in ds.coords
        if "datetime64" in str(ds[coord].dtype)
    }


def dataset_to_cases(ds: xr.Dataset) -> dict[str, dict[str, xr.DataArray]]:
    cases: dict[str, dict[str, xr.DataArray]] = {"imbalanced": {}, "balanced": {}}
    for name, da in ds.data_vars.items():
        case_name, var_name = name.split("__", 1)
        if case_name not in cases:
            cases[case_name] = {}
        time_dim = da.dims[0]
        cases[case_name][var_name] = da.rename({time_dim: "time"})
    return cases


def load_or_compute_cases(args: argparse.Namespace) -> dict[str, dict[str, xr.DataArray]]:
    cache_file = args.cache_file or default_cache_file(args.output)
    if not args.no_cache and cache_file.exists() and not args.recompute_cache:
        progress(f"Checking cache: {cache_file}")
        with xr.open_dataset(cache_file) as cached:
            if cache_matches(cached, args):
                progress(f"Using cached processed point data: {cache_file}")
                return dataset_to_cases(cached.load())
        progress(f"Cache metadata differs from requested options; recomputing {cache_file}")

    progress("Cache miss or recompute requested; reading source NetCDF files")

    imbalanced = load_case(
        args.base_dir,
        args.imbalanced_exp,
        args.month,
        args.start,
        args.end,
        args.lat,
        args.lon,
        args.time_chunk,
    )
    balanced = load_case(
        args.base_dir,
        args.balanced_exp,
        args.month,
        args.start,
        args.end,
        args.lat,
        args.lon,
        args.time_chunk,
    )
    cases = {"imbalanced": imbalanced, "balanced": balanced}

    if not args.no_cache:
        progress(f"Writing processed cache: {cache_file}")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_ds = cases_to_dataset(cases, args)
        cache_ds.to_netcdf(cache_file, encoding=cache_encoding(cache_ds, args))
        progress(f"Saved processed point cache: {cache_file}")

    return cases


# ==============================================================================
# Direct MC cache loading
# ==============================================================================
SECONDS_PER_DAY = 86400.0


def default_mc_cache_dir() -> Path:
    return Path(__file__).with_name("cache_mse_budget_wp_2wk")


def load_mc_cache(
    mc_cache_dir: Path | None,
    start: str,
    end: str,
) -> dict[str, xr.Dataset] | None:
    """Try to load direct-MC cache files for both experiments.

    Returns a dict {"imbalanced": ds, "balanced": ds} or None if not available.
    """
    cache_dir = mc_cache_dir or default_mc_cache_dir()
    imbalanced_file = cache_dir / "Reanalysis-IC_wtp_direct_mc.nc"
    balanced_file = cache_dir / "IAU-IC_wtp_direct_mc.nc"

    if not imbalanced_file.exists() or not balanced_file.exists():
        progress(
            f"MC cache not found in {cache_dir} "
            f"(need Reanalysis-IC_wtp_direct_mc.nc and IAU-IC_wtp_direct_mc.nc). "
            "Skipping MC panels. Run mse_budget_wp_2wk.py first."
        )
        return None

    progress(f"Loading direct-MC cache from {cache_dir}")
    mc = {}
    for key, path in [("imbalanced", imbalanced_file), ("balanced", balanced_file)]:
        ds = xr.open_dataset(path).load()
        ds = ds.sel(time=slice(start, end))
        mc[key] = ds
        progress(f"  {key}: {ds.sizes.get('time', 0)} time steps, vars={list(ds.data_vars)}")
    return mc


def style_time_axis(ax: plt.Axes) -> None:
    ax.set_xlabel("Date (month-day)")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    ax.grid(True, linestyle="--", alpha=0.6)


def line(ax: plt.Axes, da: xr.DataArray, **kwargs: object) -> list[object]:
    return ax.plot(da.time.values, da.values, **kwargs)


def plot_sst_air(ax: plt.Axes, case: dict[str, xr.DataArray], title: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold")
    color1 = "green"
    color2 = "darkblue"

    line1 = line(
        ax,
        case["sst_tendency"],
        color=color1,
        label="SST tendencies",
    )
    ax.set_ylabel("SST tendencies (K/3hr)", color=color1)
    ax.tick_params(axis="y", labelcolor=color1)
    ax.set_ylim(-0.1, 1.1)
    style_time_axis(ax)

    ax2 = ax.twinx()
    line2 = line(ax2, case["air_temp"], color=color2, label="Air Temp")
    ax2.set_ylabel("Air Temperature (K)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(298, 302)

    lines = line1 + line2
    ax.legend(lines, [line.get_label() for line in lines], loc="best")


def plot_heat_fluxes(ax: plt.Axes, case: dict[str, xr.DataArray], title: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold")
    color1 = "green"
    color2 = "darkblue"

    line1 = line(ax, case["shfx"], color=color1, label="Sensible Heat flux")
    ax.set_ylabel("Sensible Heat Flux (W/m^2)", color=color1)
    ax.tick_params(axis="y", labelcolor=color1)
    ax.set_ylim(2, 55)
    style_time_axis(ax)

    ax2 = ax.twinx()
    line2 = line(
        ax2,
        case["vertical_heat_flux"],
        color=color2,
        label="- T*omega500",
    )
    ax2.set_ylabel("vertical Heat flux 500 (K*Pa/s)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(-130, 320)

    lines = line1 + line2
    ax.legend(lines, [line.get_label() for line in lines], loc="best")


def plot_omega_cape_precip(ax: plt.Axes, case: dict[str, xr.DataArray], title: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold")
    color1 = "green"
    color2 = "darkblue"
    color3 = "0.45"

    line1 = line(ax, case["cape"], color=color1, label="CAPE")
    ax.set_ylabel("CAPE J kg-1", color=color1)
    ax.tick_params(axis="y", labelcolor=color1)
    ax.set_ylim(0, 2800)
    style_time_axis(ax)

    ax2 = ax.twinx()
    line2 = line(ax2, case["omega"], color=color2, label="Omega500")
    ax2.set_ylabel("Omega500", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(-1.2, 2.1)

    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("axes", 1.28))
    ax3.spines["right"].set_color(color3)
    line3 = line(
        ax3,
        case["precip"],
        color=color3,
        label="Precip",
        linewidth=1.6,
        alpha=0.55,
    )
    ax3.set_ylabel("Precip (mm day-1)", color=color3)
    ax3.tick_params(axis="y", labelcolor=color3)
    ax3.set_ylim(0, 120)

    lines = line1 + line2 + line3
    ax.legend(lines, [line.get_label() for line in lines], loc="upper right")


def plot_mc_precip(ax: plt.Axes, mc_ds: xr.Dataset, title: str) -> None:
    """Plot directly computed moisture convergence with precipitation overlay."""
    ax.set_title(title, fontsize=12, fontweight="bold")
    color_mc = "green"
    color_pr = "darkblue"

    mc_mm_day = mc_ds["MCDirect"] * SECONDS_PER_DAY
    precip_mm_day = mc_ds["Precip"] * SECONDS_PER_DAY

    line1 = line(ax, mc_mm_day, color=color_mc, label="MC (direct)")
    ax.set_ylabel("MC (mm day⁻¹)", color=color_mc)
    ax.tick_params(axis="y", labelcolor=color_mc)
    style_time_axis(ax)

    ax2 = ax.twinx()
    line2 = line(ax2, precip_mm_day, color=color_pr, label="Precip", alpha=0.55, linewidth=1.6)
    ax2.set_ylabel("Precip (mm day⁻¹)", color=color_pr)
    ax2.tick_params(axis="y", labelcolor=color_pr)

    lines = line1 + line2
    ax.legend(lines, [ln.get_label() for ln in lines], loc="upper right")


def plot_figure(
    imbalanced: dict[str, xr.DataArray],
    balanced: dict[str, xr.DataArray],
    output: Path,
    mc_data: dict[str, xr.Dataset] | None = None,
) -> None:
    has_mc = mc_data is not None
    ncols = 4 if has_mc else 3
    figw = 16 if has_mc else 12

    progress(f"Plotting figure to {output} ({'with' if has_mc else 'without'} MC panels, {ncols} columns)")
    fig = plt.figure(figsize=(figw, 6))
    fig.text(
        0.025,
        0.75,
        "Dynamically Imbalanced",
        va="center",
        ha="center",
        rotation="vertical",
        fontsize=12,
        fontweight="bold",
        color="blue",
    )
    fig.text(
        0.025,
        0.25,
        "Dynamically Balanced",
        va="center",
        ha="center",
        rotation="vertical",
        fontsize=12,
        fontweight="bold",
        color="orange",
    )

    # Row 1: imbalanced
    plot_sst_air(plt.subplot(2, ncols, 1), imbalanced, "(a) SST tendencies and Air Temp")
    plot_heat_fluxes(plt.subplot(2, ncols, 2), imbalanced, "(b) Heat Fluxes")
    plot_omega_cape_precip(plt.subplot(2, ncols, 3), imbalanced, "(c) Omega500, CAPE, and Precip")
    if has_mc:
        plot_mc_precip(plt.subplot(2, ncols, 4), mc_data["imbalanced"], "(d) MC and Precip")

    # Row 2: balanced
    bal_labels = ["(d)", "(e)", "(f)"] if not has_mc else ["(e)", "(f)", "(g)"]
    plot_sst_air(plt.subplot(2, ncols, ncols + 1), balanced, f"{bal_labels[0]} SST tendencies and Air Temp")
    plot_heat_fluxes(plt.subplot(2, ncols, ncols + 2), balanced, f"{bal_labels[1]} Heat Fluxes")
    plot_omega_cape_precip(plt.subplot(2, ncols, ncols + 3), balanced, f"{bal_labels[2]} Omega500, CAPE, and Precip")
    if has_mc:
        plot_mc_precip(plt.subplot(2, ncols, ncols + 4), mc_data["balanced"], "(h) MC and Precip")

    plt.tight_layout(rect=[0.075, 0.02, 0.98, 0.96])
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dask.config.set(scheduler="synchronous")
    progress(
        "Starting WP diagnostic: "
        f"{args.imbalanced_exp} vs {args.balanced_exp}, "
        f"{args.start} to {args.end}, lat={args.lat}, lon={args.lon}"
    )

    cases = load_or_compute_cases(args)
    mc_data = load_mc_cache(args.mc_cache_dir, args.start, args.end)
    plot_figure(cases["imbalanced"], cases["balanced"], args.output, mc_data=mc_data)
    progress(f"Saved {args.output}")


if __name__ == "__main__":
    main()
