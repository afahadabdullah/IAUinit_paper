#!/usr/bin/env python3
"""Compute ensemble-mean Wheeler-Kiladis spectra for 3-hourly precip."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.environ.get('USER', 'wk')}")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


DEFAULT_GROUPS = ("ME", "RP")
DEFAULT_MEMBER_SUFFIXES = (
    "0416",
    "0421",
    "0426",
    "0506",
    "0511",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Treat ME/RP experiments as ensemble members and plot ensemble-mean "
            "WK spectrum from 3-hourly precipitation."
        )
    )
    parser.add_argument(
        "--base-dir",
        default="/nobackupp27/afahad/exp/IAU_exp",
        help="Directory containing GEOSMIT_ME* and GEOSMIT_RP* experiment directories.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=("ME", "RP", "me", "rp"),
        default=list(DEFAULT_GROUPS),
        help="Experiment groups to process.",
    )
    parser.add_argument(
        "--members",
        nargs="+",
        default=list(DEFAULT_MEMBER_SUFFIXES),
        help=(
            "Member suffixes to process, e.g. 0416 0421. Full names like "
            "GEOSMIT_RP0416 are also accepted and mapped to each requested group."
        ),
    )
    parser.add_argument(
        "--collection",
        default="geosgcm_surf",
        help="GEOS holding collection containing PRECTOT files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.geosgcm_surf.*z.nc4",
        help="Glob pattern inside each monthly collection directory.",
    )
    parser.add_argument(
        "--var",
        default="PRECTOT",
        help="Precipitation variable name.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for PNG and NetCDF files. Default: this script's directory.",
    )
    parser.add_argument(
        "--out-prefix",
        default=None,
        help=(
            "Optional base prefix override. By default RP uses the existing "
            "wk_rp_precip_ensmean cache name and ME uses wk_me_precip_ensmean."
        ),
    )
    parser.add_argument(
        "--me-prefix",
        default="wk_me_precip_ensmean",
        help="Output prefix for the ME cache and plot.",
    )
    parser.add_argument(
        "--rp-prefix",
        default="wk_rp_precip_ensmean",
        help="Output prefix for the RP cache and plot.",
    )
    parser.add_argument(
        "--comparison-prefix",
        default="wk_me_rp_precip_ensmean",
        help="Output prefix for the combined ME/RP/ME-RP plot.",
    )
    parser.add_argument(
        "--lat-bounds",
        nargs=2,
        type=float,
        default=(-15.0, 15.0),
        metavar=("LAT_MIN", "LAT_MAX"),
        help="Latitude band for WK analysis.",
    )
    parser.add_argument(
        "--lat-step",
        type=float,
        default=1.0,
        help="Latitude spacing used if interpolation to a symmetric grid is needed.",
    )
    parser.add_argument(
        "--component",
        choices=("sym", "asym", "raw"),
        default="sym",
        help="Latitude component to analyze: symmetric, antisymmetric, or raw anomalies.",
    )
    parser.add_argument(
        "--dt-hours",
        type=float,
        default=3.0,
        help="Temporal sampling interval in hours.",
    )
    parser.add_argument(
        "--max-days",
        type=float,
        default=None,
        help="Optional maximum lead length in days. For 3-hourly data, 60 gives 480 samples.",
    )
    parser.add_argument(
        "--start-time",
        default=None,
        help="Optional calendar start time passed to xarray sel(time=slice(...)).",
    )
    parser.add_argument(
        "--end-time",
        default=None,
        help="Optional calendar end time passed to xarray sel(time=slice(...)).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=86400.0,
        help="Scale applied to PRECTOT. Default converts kg m-2 s-1 or mm s-1 to mm day-1.",
    )
    parser.add_argument(
        "--kmax",
        type=int,
        default=15,
        help="Maximum absolute zonal wavenumber to plot/save.",
    )
    parser.add_argument(
        "--fmin",
        type=float,
        default=0.01,
        help="Minimum plotted frequency in cycles per day.",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=0.8,
        help="Maximum plotted frequency in cycles per day.",
    )
    parser.add_argument(
        "--plot-period-days",
        nargs=2,
        type=float,
        default=(60.0, 3.0),
        metavar=("PERIOD_LOW", "PERIOD_HIGH"),
        help=(
            "Displayed y-axis period range in days per cycle. Default 60 3 shows "
            "the 60-day to 3-day band. Use 0 0 to show the old full frequency range."
        ),
    )
    parser.add_argument(
        "--smooth-passes",
        type=int,
        default=40,
        help="Number of 1-2-1 smoothing passes for the background spectrum.",
    )
    parser.add_argument(
        "--smooth-freq-passes",
        type=int,
        default=None,
        help=(
            "Optional number of background smoothing passes along frequency. "
            "Default: use --smooth-passes."
        ),
    )
    parser.add_argument(
        "--smooth-wave-passes",
        type=int,
        default=None,
        help=(
            "Optional number of background smoothing passes along zonal wavenumber. "
            "Default: use --smooth-passes."
        ),
    )
    parser.add_argument(
        "--background-method",
        choices=("log", "linear"),
        default="log",
        help=(
            "Background estimate for normalized plots. 'log' smooths log10(power) "
            "and usually retains sharper WK structure; 'linear' is the old method."
        ),
    )
    parser.add_argument(
        "--comparison-background",
        choices=("shared", "separate"),
        default="shared",
        help=(
            "For normalized ME/RP comparison plots, use one shared ME/RP background "
            "or separate backgrounds for each experiment. Shared averages the ME and "
            "RP backgrounds and makes the difference panel equivalent to log10(ME/RP)."
        ),
    )
    parser.add_argument(
        "--normalized-levels",
        choices=("adaptive", "fixed"),
        default="adaptive",
        help=(
            "Color levels for normalized plots. 'adaptive' uses centered percentile "
            "levels around zero; 'fixed' uses the old 0..0.6 positive-only levels."
        ),
    )
    parser.add_argument(
        "--plot-smooth-passes",
        type=int,
        default=4,
        help=(
            "Number of 1-2-1 smoothing passes applied only to plotted log/ratio "
            "fields. Use 0 to show unsmoothed spectral bins."
        ),
    )
    parser.add_argument(
        "--level-percentile",
        type=float,
        default=97.5,
        help="Percentile used for adaptive color limits in plots.",
    )
    parser.add_argument(
        "--plot-mode",
        choices=("power", "normalized", "both"),
        default="power",
        help="Plot raw log10 power, background-normalized log10(power/background), or both.",
    )
    parser.add_argument(
        "--no-truncate-to-common",
        action="store_true",
        help="Do not truncate all members to the shortest available lead length.",
    )
    parser.add_argument(
        "--no-detrend",
        action="store_true",
        help="Skip linear detrending before the FFT.",
    )
    parser.add_argument(
        "--no-taper",
        action="store_true",
        help="Skip the Hanning taper in time before the FFT.",
    )
    parser.add_argument(
        "--plot-members",
        action="store_true",
        help="Also save one WK plot per ensemble member.",
    )
    parser.add_argument(
        "--skip-group-plots",
        action="store_true",
        help="Only save the combined ME/RP comparison plot.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Only plot from existing NetCDF caches; fail instead of reading raw files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore an existing cached NetCDF spectrum and recompute from the source files.",
    )
    return parser.parse_args()


def normalized_groups(groups: list[str]) -> list[str]:
    seen = set()
    out = []
    for group in groups:
        normalized = group.upper()
        if normalized not in seen:
            out.append(normalized)
            seen.add(normalized)
    return out


def group_member_names(group: str, members: list[str]) -> list[str]:
    names = []
    for member in members:
        suffix = member
        if suffix.startswith("GEOSMIT_"):
            suffix = suffix.split("_", 1)[1]
        if suffix.startswith(("ME", "RP")):
            suffix = suffix[2:]
        names.append(f"GEOSMIT_{group}{suffix}")
    return names


def output_prefix_for_group(args: argparse.Namespace, group: str) -> str:
    group_lower = group.lower()
    base_prefix = args.out_prefix
    if base_prefix is None:
        return args.me_prefix if group == "ME" else args.rp_prefix
    if "{group}" in base_prefix:
        return base_prefix.format(group=group_lower)
    if base_prefix.startswith(f"wk_{group_lower}_"):
        return base_prefix
    return f"{base_prefix}_{group_lower}"


def comparison_prefix(args: argparse.Namespace, groups: list[str]) -> str:
    if args.comparison_prefix:
        return args.comparison_prefix
    lower_groups = [group.lower() for group in groups]
    if "me" in lower_groups and "rp" in lower_groups:
        return "wk_me_rp_precip_ensmean"
    return f"wk_{'_'.join(lower_groups)}_precip_ensmean"


def selected_plot_modes(args: argparse.Namespace) -> list[str]:
    if args.plot_mode == "both":
        return ["power", "normalized"]
    return [args.plot_mode]


def background_smooth_passes(args: argparse.Namespace) -> tuple[int, int]:
    freq_passes = args.smooth_freq_passes
    wave_passes = args.smooth_wave_passes
    if freq_passes is None:
        freq_passes = args.smooth_passes
    if wave_passes is None:
        wave_passes = args.smooth_passes
    return max(0, int(freq_passes)), max(0, int(wave_passes))


def plot_path(out_dir: Path, prefix: str, plot_mode: str, n_modes: int) -> Path:
    if n_modes > 1:
        return out_dir / f"{prefix}_{plot_mode}.png"
    return out_dir / f"{prefix}.png"


def member_files(base_dir: Path, member: str, collection: str, pattern: str) -> list[Path]:
    root = base_dir / member / "holding" / collection
    files = sorted(root.glob(f"*/{pattern}"))
    if not files:
        files = sorted(root.rglob(pattern))
    files = [p for p in files if "monthly" not in p.name]
    return files


def time_count(
    files: list[Path],
    var_name: str,
    start_time: str | None,
    end_time: str | None,
) -> int:
    count = 0
    for path in files:
        try:
            with xr.open_dataset(path) as ds:
                if var_name not in ds:
                    print(f"  warning: {var_name} missing from {path}")
                    continue
                da = ds[var_name]
                if "time" not in da.dims:
                    raise ValueError(f"{path} has no time dimension for {var_name}")
                if start_time is not None or end_time is not None:
                    da = da.sel(time=slice(start_time, end_time))
                count += int(da.sizes["time"])
        except Exception as exc:
            print(f"  warning: cannot count times in {path}: {exc}")
    return count


def normalize_dims(da: xr.DataArray) -> xr.DataArray:
    renames = {}
    for name in da.dims:
        lname = name.lower()
        if lname in {"latitude", "lat"} and name != "lat":
            renames[name] = "lat"
        if lname in {"longitude", "lon"} and name != "lon":
            renames[name] = "lon"
    if renames:
        da = da.rename(renames)

    missing = {"time", "lat", "lon"} - set(da.dims)
    if missing:
        raise ValueError(f"missing required dimensions: {sorted(missing)}")
    return da.transpose("time", "lat", "lon")


def load_precip_member(
    files: list[Path],
    var_name: str,
    max_steps: int | None,
    start_time: str | None,
    end_time: str | None,
    scale: float,
) -> xr.DataArray:
    parts = []
    remaining = max_steps
    for index, path in enumerate(files, start=1):
        if remaining is not None and remaining <= 0:
            break
        try:
            with xr.open_dataset(path) as ds:
                if var_name not in ds:
                    print(f"  warning: {var_name} missing from {path}")
                    continue
                da = normalize_dims(ds[var_name])
                if start_time is not None or end_time is not None:
                    da = da.sel(time=slice(start_time, end_time))
                if da.sizes["time"] == 0:
                    continue
                if remaining is not None and da.sizes["time"] > remaining:
                    da = da.isel(time=slice(0, remaining))
                da = (da * scale).load()
                da.attrs["units"] = "mm/day"
                parts.append(da)
                if remaining is not None:
                    remaining -= int(da.sizes["time"])
        except Exception as exc:
            print(f"  warning: skipping {path}: {exc}")
        if index % 100 == 0:
            print(f"  loaded {index}/{len(files)} files")

    if not parts:
        raise RuntimeError(f"no usable {var_name} data found")

    da = xr.concat(parts, dim="time").sortby("time")
    if max_steps is not None:
        da = da.isel(time=slice(0, max_steps))
        if da.sizes["time"] < max_steps:
            raise RuntimeError(
                f"only loaded {da.sizes['time']} samples, but {max_steps} were requested"
            )
    return da


def select_tropics(da: xr.DataArray, lat_bounds: tuple[float, float]) -> xr.DataArray:
    lat_min, lat_max = sorted(lat_bounds)
    da = da.sortby("lat")
    if float(da.lon.min().values) < 0.0:
        da = da.assign_coords(lon=(da.lon + 360.0) % 360.0).sortby("lon")
    return da.sel(lat=slice(lat_min, lat_max))


def symmetric_grid(
    anom: xr.DataArray,
    lat_bounds: tuple[float, float],
    lat_step: float,
) -> xr.DataArray:
    lats = anom.lat.values
    if np.allclose(lats, -lats[::-1], atol=1.0e-6):
        return anom

    lat_min, lat_max = sorted(lat_bounds)
    new_lats = np.arange(lat_min, lat_max + 0.5 * lat_step, lat_step)
    print(f"  interpolating latitude to symmetric grid: {new_lats[0]}..{new_lats[-1]}")
    return anom.interp(lat=new_lats)


def component_array(
    da: xr.DataArray,
    lat_bounds: tuple[float, float],
    lat_step: float,
    component: str,
) -> xr.DataArray:
    anom = da - da.mean("time")
    if component == "raw":
        return anom.fillna(0.0)

    anom = symmetric_grid(anom, lat_bounds, lat_step).fillna(0.0)
    values = anom.values
    flipped = values[:, ::-1, :]
    if component == "sym":
        out = 0.5 * (values + flipped)
        name = "symmetric_precip_anom"
    else:
        out = 0.5 * (values - flipped)
        name = "antisymmetric_precip_anom"
    return xr.DataArray(out, coords=anom.coords, dims=anom.dims, name=name)


def detrend_time(data: np.ndarray) -> np.ndarray:
    nt = data.shape[0]
    x = np.arange(nt, dtype=np.float64)
    x = x - x.mean()
    denom = np.sum(x * x)
    if denom == 0.0:
        return data
    slope = np.tensordot(x, data, axes=(0, 0)) / denom
    intercept = data.mean(axis=0)
    return data - (intercept[None, :, :] + x[:, None, None] * slope[None, :, :])


def wk_power(
    da: xr.DataArray,
    dt_days: float,
    kmax: int,
    fmin: float,
    fmax: float,
    detrend: bool,
    taper: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = da.values.astype(np.float64, copy=True)
    if detrend:
        data = detrend_time(data)
    else:
        data = data - data.mean(axis=0, keepdims=True)
    if taper:
        data = data * np.hanning(data.shape[0])[:, None, None]

    nt, _, nlon = data.shape
    fft_out = np.fft.fft2(data, axes=(0, 2))
    power = np.abs(fft_out) ** 2
    power_lat_sum = np.sum(power, axis=1)

    freqs = np.fft.fftfreq(nt, d=dt_days)
    ks = np.fft.fftfreq(nlon, d=1.0 / nlon)
    power_shifted = np.fft.fftshift(power_lat_sum, axes=1)
    ks_shifted = np.fft.fftshift(ks)

    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    k_mask = (ks_shifted >= -kmax) & (ks_shifted <= kmax)
    return power_shifted[np.ix_(freq_mask, k_mask)], freqs[freq_mask], ks_shifted[k_mask]


def smooth121(arr: np.ndarray, axis: int) -> np.ndarray:
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (1, 1)
    padded = np.pad(arr, pad_width, mode="edge")
    center = [slice(None)] * arr.ndim
    lower = [slice(None)] * arr.ndim
    upper = [slice(None)] * arr.ndim
    center[axis] = slice(1, -1)
    lower[axis] = slice(0, -2)
    upper[axis] = slice(2, None)
    return (
        0.25 * padded[tuple(lower)]
        + 0.50 * padded[tuple(center)]
        + 0.25 * padded[tuple(upper)]
    )


def background_spectrum(
    power: np.ndarray,
    freq_passes: int,
    wave_passes: int,
    method: str = "log",
) -> np.ndarray:
    power_safe = np.maximum(power, np.finfo(np.float64).tiny)
    if method == "log":
        bg = np.log10(power_safe)
    else:
        bg = power_safe.astype(np.float64, copy=True)

    for _ in range(max(0, freq_passes)):
        bg = smooth121(bg, axis=0)
    for _ in range(max(0, wave_passes)):
        bg = smooth121(bg, axis=1)

    if method == "log":
        bg = 10.0**bg
    return np.maximum(bg, np.finfo(np.float64).tiny)


def add_dispersion_curves(ax: plt.Axes, fmin: float, fmax: float) -> None:
    radius = 6.371e6
    omega_earth = 7.2921e-5
    gravity = 9.8
    beta = 2.0 * omega_earth / radius

    for depth in (8, 25, 90):
        c = np.sqrt(gravity * depth)
        kk = np.linspace(0.1, 15.0, 200)
        ff = 86400.0 * c * kk / (2.0 * np.pi * radius)
        mask = (ff >= fmin) & (ff <= fmax)
        if np.any(mask):
            ax.plot(kk[mask], ff[mask], color="k", linewidth=1.0)
            ax.text(kk[mask][-1], ff[mask][-1], f"{depth}m", fontsize=8)

    for depth in (8, 25, 90):
        c = np.sqrt(gravity * depth)
        kk = np.linspace(-15.0, -0.1, 200)
        kk_rad = kk / radius
        omega = -beta * kk_rad / (kk_rad**2 + 3.0 * beta / c)
        ff = 86400.0 * omega / (2.0 * np.pi)
        mask = (ff >= fmin) & (ff <= fmax)
        if np.any(mask):
            ax.plot(kk[mask], ff[mask], color="k", linewidth=1.0, linestyle="--")
            ax.text(kk[mask][0], ff[mask][0], f"{depth}m", fontsize=8)


def plot_values(
    power: np.ndarray,
    plot_mode: str,
    smooth_freq_passes: int,
    smooth_wave_passes: int,
    background_method: str,
) -> tuple[np.ndarray, str, str]:
    if plot_mode == "normalized":
        bg = background_spectrum(
            power, smooth_freq_passes, smooth_wave_passes, background_method
        )
        values = np.log10(np.maximum(power, np.finfo(np.float64).tiny) / bg)
        return (
            values,
            "log10(power / background)",
            (
                f"background normalized, {background_method} background, "
                f"{smooth_freq_passes}x{smooth_wave_passes} smooth"
            ),
        )

    values = np.log10(np.maximum(power, np.finfo(np.float64).tiny))
    return values, "log10(power)", "raw power"


def normalized_with_background(power: np.ndarray, background: np.ndarray) -> np.ndarray:
    power_safe = np.maximum(power, np.finfo(np.float64).tiny)
    bg_safe = np.maximum(background, np.finfo(np.float64).tiny)
    return np.log10(power_safe / bg_safe)


def log_power_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    num_safe = np.maximum(numerator, np.finfo(np.float64).tiny)
    den_safe = np.maximum(denominator, np.finfo(np.float64).tiny)
    return np.log10(num_safe / den_safe)


def average_background(
    me_power: np.ndarray,
    rp_power: np.ndarray,
    smooth_freq_passes: int,
    smooth_wave_passes: int,
    background_method: str,
) -> np.ndarray:
    me_bg = background_spectrum(
        me_power, smooth_freq_passes, smooth_wave_passes, background_method
    )
    rp_bg = background_spectrum(
        rp_power, smooth_freq_passes, smooth_wave_passes, background_method
    )
    return 0.5 * (me_bg + rp_bg)


def percentile_levels(values: np.ndarray, low: float, high: float, count: int) -> np.ndarray:
    vmin = np.nanpercentile(values, low)
    vmax = np.nanpercentile(values, high)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return np.linspace(vmin, vmax, count)


def centered_percentile_levels(values: np.ndarray, percentile: float, count: int) -> np.ndarray:
    limit = np.nanpercentile(np.abs(values), percentile)
    if not np.isfinite(limit) or np.isclose(limit, 0.0):
        limit = float(np.nanmax(np.abs(values)))
    if np.isclose(limit, 0.0):
        limit = 1.0
    return np.linspace(-limit, limit, count)


def frequency_to_period(freq: np.ndarray | float) -> np.ndarray | float:
    freq_arr = np.asarray(freq, dtype=np.float64)
    return 1.0 / np.maximum(freq_arr, np.finfo(np.float64).tiny)


def period_to_frequency(period: np.ndarray | float) -> np.ndarray | float:
    period_arr = np.asarray(period, dtype=np.float64)
    return 1.0 / np.maximum(period_arr, np.finfo(np.float64).tiny)


def plot_frequency_limits(
    freqs: np.ndarray,
    plot_period_days: tuple[float, float] | list[float],
) -> tuple[float, float]:
    data_low = float(freqs.min())
    data_high = float(freqs.max())
    periods = np.asarray(plot_period_days, dtype=np.float64)
    if periods.size == 2 and np.all(periods > 0.0):
        freq_low = 1.0 / float(np.max(periods))
        freq_high = 1.0 / float(np.min(periods))
    else:
        freq_low = data_low
        freq_high = min(0.35, data_high)

    freq_low = max(freq_low, data_low)
    freq_high = min(freq_high, data_high)
    if not np.isfinite(freq_low) or not np.isfinite(freq_high) or freq_low >= freq_high:
        return data_low, min(0.35, data_high)
    return freq_low, freq_high


def values_in_frequency_range(
    values: np.ndarray,
    freqs: np.ndarray,
    ylim: tuple[float, float],
) -> np.ndarray:
    mask = (freqs >= ylim[0]) & (freqs <= ylim[1])
    if not np.any(mask):
        return values
    return values[mask, :]


def add_period_axis(ax: plt.Axes, label: bool = True) -> None:
    secax = ax.secondary_yaxis(
        "right", functions=(frequency_to_period, period_to_frequency)
    )
    if label:
        secax.set_ylabel("Period (days per cycle)")
    ymin, ymax = ax.get_ylim()
    period_low = min(float(frequency_to_period(ymin)), float(frequency_to_period(ymax)))
    period_high = max(float(frequency_to_period(ymin)), float(frequency_to_period(ymax)))
    candidate_ticks = np.array(
        [90, 60, 45, 30, 20, 15, 12, 10, 8, 6, 5, 4, 3, 2, 1],
        dtype=float,
    )
    ticks = candidate_ticks[
        (candidate_ticks >= period_low - 1.0e-9)
        & (candidate_ticks <= period_high + 1.0e-9)
    ]
    if ticks.size:
        secax.set_yticks(ticks)
        secax.set_yticklabels([f"{tick:g}" for tick in ticks])


def smooth_plot_field(values: np.ndarray, passes: int) -> np.ndarray:
    smoothed = values.astype(np.float64, copy=True)
    for _ in range(max(0, passes)):
        smoothed = smooth121(smoothed, axis=0)
        smoothed = smooth121(smoothed, axis=1)
    return smoothed


def plot_wk(
    power: np.ndarray,
    freqs: np.ndarray,
    ks: np.ndarray,
    output_file: Path,
    title: str,
    smooth_freq_passes: int,
    smooth_wave_passes: int,
    plot_mode: str,
    background_method: str,
    normalized_levels: str,
    plot_smooth_passes: int,
    level_percentile: float,
    plot_period_days: tuple[float, float] | list[float],
) -> None:
    signal, colorbar_label, title_extra = plot_values(
        power, plot_mode, smooth_freq_passes, smooth_wave_passes, background_method
    )
    signal = smooth_plot_field(signal, plot_smooth_passes)
    ylim = plot_frequency_limits(freqs, plot_period_days)
    visible_signal = values_in_frequency_range(signal, freqs, ylim)
    if plot_mode == "normalized":
        if normalized_levels == "fixed":
            levels = np.linspace(0.0, 0.6, 13)
            cmap = "YlOrRd"
            extend = "max"
        else:
            levels = centered_percentile_levels(visible_signal, level_percentile, 21)
            cmap = "RdBu_r"
            extend = "both"
    else:
        low = max(0.0, 100.0 - level_percentile)
        levels = percentile_levels(visible_signal, low, level_percentile, 21)
        cmap = "Spectral_r"
        extend = "both"

    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    mesh = ax.contourf(ks, freqs, signal, levels=levels, cmap=cmap, extend=extend)
    fig.colorbar(mesh, ax=ax, label=colorbar_label, pad=0.12)
    ax.axvline(0.0, color="k", linestyle="--", linewidth=0.8)
    add_dispersion_curves(ax, ylim[0], ylim[1])
    ax.set_title(f"{title} ({title_extra})")
    ax.set_xlabel("Zonal wavenumber (eastward > 0)")
    ax.set_ylabel("Frequency (cycles day$^{-1}$)")
    ax.set_xlim(float(ks.min()), float(ks.max()))
    ax.set_ylim(*ylim)
    add_period_axis(ax)
    ax.grid(alpha=0.25)

    fig.subplots_adjust(left=0.09, right=0.86, bottom=0.10, top=0.92)
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_me_rp_comparison(
    me_result: dict[str, object],
    rp_result: dict[str, object],
    output_file: Path,
    args: argparse.Namespace,
    plot_mode: str,
) -> None:
    freqs = me_result["freqs"]
    ks = me_result["ks"]
    if not (
        np.allclose(freqs, rp_result["freqs"])
        and np.allclose(ks, rp_result["ks"])
        and me_result["power_ens"].shape == rp_result["power_ens"].shape
    ):
        raise ValueError("ME and RP spectra do not share the same frequency/wavenumber grid")

    me_power = me_result["power_ens"]
    rp_power = rp_result["power_ens"]
    smooth_freq_passes, smooth_wave_passes = background_smooth_passes(args)
    ylim = plot_frequency_limits(freqs, args.plot_period_days)
    if plot_mode == "normalized" and args.comparison_background == "shared":
        shared_bg = average_background(
            me_power,
            rp_power,
            smooth_freq_passes,
            smooth_wave_passes,
            args.background_method,
        )
        me_values = normalized_with_background(me_power, shared_bg)
        rp_values = normalized_with_background(rp_power, shared_bg)
        colorbar_label = "log10(power / mean ME/RP background)"
        title_extra = (
            f"mean ME/RP {args.background_method} background, "
            f"{smooth_freq_passes}x{smooth_wave_passes} smooth"
        )
    else:
        me_values, colorbar_label, title_extra = plot_values(
            me_power,
            plot_mode,
            smooth_freq_passes,
            smooth_wave_passes,
            args.background_method,
        )
        rp_values, _, _ = plot_values(
            rp_power,
            plot_mode,
            smooth_freq_passes,
            smooth_wave_passes,
            args.background_method,
        )
    me_values = smooth_plot_field(me_values, args.plot_smooth_passes)
    rp_values = smooth_plot_field(rp_values, args.plot_smooth_passes)
    combined = np.concatenate(
        [
            values_in_frequency_range(me_values, freqs, ylim).ravel(),
            values_in_frequency_range(rp_values, freqs, ylim).ravel(),
        ]
    )
    if plot_mode == "normalized":
        if args.normalized_levels == "fixed":
            common_levels = np.linspace(0.0, 0.6, 13)
            common_cmap = "YlOrRd"
            common_extend = "max"
        else:
            common_levels = centered_percentile_levels(combined, args.level_percentile, 21)
            common_cmap = "RdBu_r"
            common_extend = "both"
    else:
        low = max(0.0, 100.0 - args.level_percentile)
        common_levels = percentile_levels(combined, low, args.level_percentile, 21)
        common_cmap = "Spectral_r"
        common_extend = "both"

    if plot_mode == "normalized":
        if args.comparison_background == "shared":
            diff = log_power_ratio(me_power, rp_power)
            diff = smooth_plot_field(diff, args.plot_smooth_passes)
            diff_label = "log10(ME / RP)"
        else:
            diff = me_values - rp_values
            diff_label = "ME - RP log10(power / background)"
    else:
        diff = me_power - rp_power
        diff = smooth_plot_field(diff, args.plot_smooth_passes)
        diff_label = "ME - RP power"
    visible_diff = values_in_frequency_range(diff, freqs, ylim)
    diff_levels = centered_percentile_levels(visible_diff, args.level_percentile, 21)

    fig, axes = plt.subplots(1, 3, figsize=(19.5, 5.8), sharey=True)
    panels = (
        ("ME", me_values, common_levels, common_cmap, common_extend, colorbar_label),
        ("RP", rp_values, common_levels, common_cmap, common_extend, colorbar_label),
        ("ME - RP", diff, diff_levels, "RdBu_r", "both", diff_label),
    )

    for ax, (title, values, levels, cmap, extend, cbar_label) in zip(axes, panels):
        mesh = ax.contourf(ks, freqs, values, levels=levels, cmap=cmap, extend=extend)
        cbar_pad = 0.12 if title == "ME - RP" else 0.04
        fig.colorbar(mesh, ax=ax, label=cbar_label, pad=cbar_pad)
        ax.axvline(0.0, color="k", linestyle="--", linewidth=0.8)
        add_dispersion_curves(ax, ylim[0], ylim[1])
        ax.set_title(f"{title} ({title_extra})" if title != "ME - RP" else title)
        ax.set_xlabel("Zonal wavenumber (eastward > 0)")
        ax.set_xlim(float(ks.min()), float(ks.max()))
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Frequency (cycles day$^{-1}$)")
    add_period_axis(axes[-1])
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.14, top=0.90, wspace=0.30)
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_wk_dataset(
    output_file: Path,
    group: str,
    members: list[str],
    member_power: np.ndarray,
    freqs: np.ndarray,
    ks: np.ndarray,
    n_time_samples: int | None,
    args: argparse.Namespace,
) -> None:
    power_mean = member_power.mean(axis=0)
    smooth_freq_passes, smooth_wave_passes = background_smooth_passes(args)
    bg = background_spectrum(
        power_mean, smooth_freq_passes, smooth_wave_passes, args.background_method
    )
    signal = np.log10(np.maximum(power_mean, np.finfo(np.float64).tiny) / bg)
    ds = xr.Dataset(
        data_vars={
            "power": (("member", "frequency", "zonal_wavenumber"), member_power),
            "power_ens_mean": (("frequency", "zonal_wavenumber"), power_mean),
            "background_ens_mean": (("frequency", "zonal_wavenumber"), bg),
            "signal_to_background": (("frequency", "zonal_wavenumber"), signal),
        },
        coords={
            "member": members,
            "frequency": freqs,
            "zonal_wavenumber": ks,
        },
        attrs={
            "description": f"{group} ensemble-mean WK spectrum from 3-hourly precipitation",
            "experiment_group": group,
            "base_dir": str(args.base_dir),
            "collection": args.collection,
            "pattern": args.pattern,
            "variable": args.var,
            "component": args.component,
            "lat_bounds": f"{args.lat_bounds[0]} {args.lat_bounds[1]}",
            "lat_step": args.lat_step,
            "dt_hours": args.dt_hours,
            "scale": args.scale,
            "kmax": args.kmax,
            "fmin": args.fmin,
            "fmax": args.fmax,
            "max_days": "None" if args.max_days is None else args.max_days,
            "truncate_to_common": str(not args.no_truncate_to_common),
            "n_time_samples": -1 if n_time_samples is None else n_time_samples,
            "detrended": str(not args.no_detrend),
            "time_taper": str(not args.no_taper),
            "background_method": args.background_method,
            "background_smooth_freq_passes": smooth_freq_passes,
            "background_smooth_wave_passes": smooth_wave_passes,
        },
    )
    ds.to_netcdf(output_file)


def attr_matches(ds: xr.Dataset, key: str, expected: str | float | int) -> bool:
    actual = ds.attrs.get(key)
    if actual is None:
        return False
    if isinstance(expected, float):
        try:
            return bool(np.isclose(float(actual), expected))
        except (TypeError, ValueError):
            return False
    return str(actual) == str(expected)


def cache_mismatch_reasons(
    ds: xr.Dataset,
    args: argparse.Namespace,
    group: str,
    members: list[str],
) -> list[str]:
    reasons = []
    required_vars = {"power", "power_ens_mean", "frequency", "zonal_wavenumber"}
    missing_vars = required_vars - set(ds.variables)
    if missing_vars:
        reasons.append(f"missing variables {sorted(missing_vars)}")

    checks: dict[str, str | float | int] = {
        "base_dir": str(args.base_dir),
        "collection": args.collection,
        "pattern": args.pattern,
        "variable": args.var,
        "component": args.component,
        "lat_bounds": f"{args.lat_bounds[0]} {args.lat_bounds[1]}",
        "lat_step": args.lat_step,
        "dt_hours": args.dt_hours,
        "scale": args.scale,
        "kmax": args.kmax,
        "fmin": args.fmin,
        "fmax": args.fmax,
        "max_days": "None" if args.max_days is None else args.max_days,
        "truncate_to_common": str(not args.no_truncate_to_common),
        "detrended": str(not args.no_detrend),
        "time_taper": str(not args.no_taper),
    }
    for key, expected in checks.items():
        if not attr_matches(ds, key, expected):
            reasons.append(f"{key} differs")

    cached_group = ds.attrs.get("experiment_group")
    if cached_group is not None and str(cached_group) != group:
        reasons.append("experiment_group differs")

    if "member" in ds.coords:
        cached_members = [str(member) for member in ds.member.values]
        missing_members = [member for member in members if member not in cached_members]
        if missing_members:
            reasons.append(f"requested members missing from cache: {missing_members}")
    else:
        reasons.append("member coordinate missing")
    return reasons


def load_cached_spectrum(
    cache_file: Path,
    args: argparse.Namespace,
    group: str,
    members: list[str],
) -> dict[str, object] | None:
    if args.force or not cache_file.exists():
        return None

    try:
        with xr.open_dataset(cache_file) as ds:
            reasons = cache_mismatch_reasons(ds, args, group, members)
            if reasons:
                print(f"Cache exists but will be recomputed: {', '.join(reasons)}")
                return None

            print(f"Using cached WK spectra: {cache_file}")
            member_power = ds["power"].sel(member=members).values.copy()
            return {
                "group": group,
                "members": members,
                "member_power": member_power,
                "power_ens": member_power.mean(axis=0),
                "freqs": ds["frequency"].values.copy(),
                "ks": ds["zonal_wavenumber"].values.copy(),
                "cache_file": cache_file,
            }
    except Exception as exc:
        print(f"Cache exists but could not be used: {exc}")
        return None


def plot_group_spectra(
    result: dict[str, object],
    out_dir: Path,
    prefix: str,
    args: argparse.Namespace,
) -> list[Path]:
    if args.skip_group_plots:
        return []

    group = result["group"]
    plot_modes = selected_plot_modes(args)
    smooth_freq_passes, smooth_wave_passes = background_smooth_passes(args)
    written = []
    for mode in plot_modes:
        png_file = plot_path(out_dir, prefix, mode, len(plot_modes))
        plot_wk(
            result["power_ens"],
            result["freqs"],
            result["ks"],
            png_file,
            f"{group} ensemble mean {args.component} {args.var} WK spectrum",
            smooth_freq_passes,
            smooth_wave_passes,
            mode,
            args.background_method,
            args.normalized_levels,
            args.plot_smooth_passes,
            args.level_percentile,
            args.plot_period_days,
        )
        written.append(png_file)
        if args.plot_members:
            for index, member in enumerate(result["members"]):
                member_prefix = f"{prefix}_{member}"
                plot_wk(
                    result["member_power"][index],
                    result["freqs"],
                    result["ks"],
                    plot_path(out_dir, member_prefix, mode, len(plot_modes)),
                    f"{member} {args.component} {args.var} WK spectrum",
                    smooth_freq_passes,
                    smooth_wave_passes,
                    mode,
                    args.background_method,
                    args.normalized_levels,
                    args.plot_smooth_passes,
                    args.level_percentile,
                    args.plot_period_days,
                )
    return written


def compute_or_load_group(
    group: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> tuple[dict[str, object], str, Path]:
    base_dir = Path(args.base_dir)
    members = group_member_names(group, args.members)
    prefix = output_prefix_for_group(args, group)
    nc_file = out_dir / f"{prefix}.nc4"
    cached = load_cached_spectrum(nc_file, args, group, members)
    if cached is not None:
        plot_files = plot_group_spectra(cached, out_dir, prefix, args)
        for plot_file in plot_files:
            print(f"  Plot:   {plot_file}")
        return cached, prefix, plot_files
    if args.cache_only:
        raise RuntimeError(f"Cache unavailable for {group}: {nc_file}")

    print(f"\nDiscovering {group} member files...")
    files_by_member = {
        member: member_files(base_dir, member, args.collection, args.pattern)
        for member in members
    }
    for member, files in files_by_member.items():
        print(f"  {member}: {len(files)} files")
    missing = [member for member, files in files_by_member.items() if not files]
    if missing:
        raise FileNotFoundError(f"no files found for: {', '.join(missing)}")

    max_steps = None
    if args.max_days is not None:
        max_steps = int(round(args.max_days * 24.0 / args.dt_hours))
        print(f"Using first {max_steps} samples from each {group} member.")
    elif not args.no_truncate_to_common:
        print(f"Counting {group} time samples...")
        counts = {
            member: time_count(files, args.var, args.start_time, args.end_time)
            for member, files in files_by_member.items()
        }
        for member, count in counts.items():
            print(f"  {member}: {count} samples")
        empty_members = [member for member, count in counts.items() if count == 0]
        if empty_members:
            raise RuntimeError(f"no usable time samples for: {', '.join(empty_members)}")
        max_steps = min(counts.values())
    if max_steps is not None and max_steps < 2:
        raise RuntimeError(f"need at least 2 time samples, got {max_steps}")

    dt_days = args.dt_hours / 24.0
    member_names = []
    spectra = []
    freqs_ref = None
    ks_ref = None

    for member, files in files_by_member.items():
        print(f"\nProcessing {member}...")
        precip = load_precip_member(
            files,
            var_name=args.var,
            max_steps=max_steps,
            start_time=args.start_time,
            end_time=args.end_time,
            scale=args.scale,
        )
        precip = select_tropics(precip, tuple(args.lat_bounds))
        comp = component_array(precip, tuple(args.lat_bounds), args.lat_step, args.component)
        power, freqs, ks = wk_power(
            comp,
            dt_days=dt_days,
            kmax=args.kmax,
            fmin=args.fmin,
            fmax=args.fmax,
            detrend=not args.no_detrend,
            taper=not args.no_taper,
        )

        if freqs_ref is None:
            freqs_ref = freqs
            ks_ref = ks
        elif not (
            freqs_ref.shape == freqs.shape
            and ks_ref.shape == ks.shape
            and np.allclose(freqs_ref, freqs)
            and np.allclose(ks_ref, ks)
        ):
            raise ValueError(
                f"{member} spectrum grid differs; use --max-days or truncate to a common length"
            )

        member_names.append(member)
        spectra.append(power)

    member_power = np.stack(spectra, axis=0)
    power_ens = member_power.mean(axis=0)
    save_wk_dataset(
        nc_file,
        group,
        member_names,
        member_power,
        freqs_ref,
        ks_ref,
        max_steps,
        args,
    )
    result = {
        "group": group,
        "members": member_names,
        "member_power": member_power,
        "power_ens": power_ens,
        "freqs": freqs_ref,
        "ks": ks_ref,
        "cache_file": nc_file,
    }
    plot_files = plot_group_spectra(result, out_dir, prefix, args)
    print(f"\nDone with {group}.")
    print(f"  NetCDF: {nc_file}")
    for plot_file in plot_files:
        print(f"  Plot:   {plot_file}")
    return result, prefix, plot_files


def main() -> None:
    args = parse_args()
    groups = normalized_groups(args.groups)
    out_dir = Path(args.out_dir) if args.out_dir is not None else Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for group in groups:
        result, _, _ = compute_or_load_group(group, args, out_dir)
        results[group] = result

    if "ME" in results and "RP" in results:
        prefix = comparison_prefix(args, groups)
        plot_modes = selected_plot_modes(args)
        for mode in plot_modes:
            compare_file = plot_path(out_dir, prefix, mode, len(plot_modes))
            plot_me_rp_comparison(results["ME"], results["RP"], compare_file, args, mode)
            print(f"\nComparison plot: {compare_file}")


if __name__ == "__main__":
    main()
