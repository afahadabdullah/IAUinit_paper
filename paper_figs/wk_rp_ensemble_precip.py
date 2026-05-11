#!/usr/bin/env python3
"""Compute an ensemble-mean Wheeler-Kiladis spectrum for RP 3-hourly precip."""

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


DEFAULT_MEMBERS = (
    "GEOSMIT_RP0416",
    "GEOSMIT_RP0421",
    "GEOSMIT_RP0426",
    "GEOSMIT_RP0506",
    "GEOSMIT_RP0511",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Treat RP experiments as ensemble members and plot the ensemble-mean "
            "WK spectrum from 3-hourly precipitation."
        )
    )
    parser.add_argument(
        "--base-dir",
        default="/nobackupp27/afahad/exp/IAU_exp",
        help="Directory containing GEOSMIT_RP* experiment directories.",
    )
    parser.add_argument(
        "--members",
        nargs="+",
        default=list(DEFAULT_MEMBERS),
        help="Experiment directories to treat as ensemble members.",
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
        default="wk_rp_precip_ensmean",
        help="Prefix for output files.",
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
        "--smooth-passes",
        type=int,
        default=10,
        help="Number of 1-2-1 smoothing passes for the background spectrum.",
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
        "--force",
        action="store_true",
        help="Ignore an existing cached NetCDF spectrum and recompute from the source files.",
    )
    return parser.parse_args()


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


def background_spectrum(power: np.ndarray, passes: int) -> np.ndarray:
    bg = power.astype(np.float64, copy=True)
    for _ in range(passes):
        bg = smooth121(bg, axis=0)
        bg = smooth121(bg, axis=1)
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


def plot_wk(
    power: np.ndarray,
    freqs: np.ndarray,
    ks: np.ndarray,
    output_file: Path,
    title: str,
    smooth_passes: int,
) -> None:
    bg = background_spectrum(power, smooth_passes)
    signal = np.log10(np.maximum(power, np.finfo(np.float64).tiny) / bg)

    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    levels = np.linspace(0.0, 0.6, 13)
    mesh = ax.contourf(ks, freqs, signal, levels=levels, cmap="YlOrRd", extend="max")
    fig.colorbar(mesh, ax=ax, label="log10(power / background)")
    ax.axvline(0.0, color="k", linestyle="--", linewidth=0.8)
    add_dispersion_curves(ax, float(freqs.min()), float(freqs.max()))
    ax.set_title(title)
    ax.set_xlabel("Zonal wavenumber (eastward > 0)")
    ax.set_ylabel("Frequency (cycles day$^{-1}$)")
    ax.set_xlim(float(ks.min()), float(ks.max()))
    ax.set_ylim(float(freqs.min()), min(0.35, float(freqs.max())))
    ax.grid(alpha=0.25)

    fig.subplots_adjust(left=0.09, right=0.88, bottom=0.10, top=0.92)
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def save_wk_dataset(
    output_file: Path,
    members: list[str],
    member_power: np.ndarray,
    freqs: np.ndarray,
    ks: np.ndarray,
    n_time_samples: int | None,
    args: argparse.Namespace,
) -> None:
    power_mean = member_power.mean(axis=0)
    bg = background_spectrum(power_mean, args.smooth_passes)
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
            "description": "RP ensemble-mean WK spectrum from 3-hourly precipitation",
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


def cache_mismatch_reasons(ds: xr.Dataset, args: argparse.Namespace) -> list[str]:
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

    if "member" in ds.coords:
        cached_members = [str(member) for member in ds.member.values]
        if cached_members != list(args.members):
            reasons.append("member list differs")
    else:
        reasons.append("member coordinate missing")
    return reasons


def plot_from_cache(cache_file: Path, out_dir: Path, args: argparse.Namespace) -> bool:
    if args.force or not cache_file.exists():
        return False

    try:
        with xr.open_dataset(cache_file) as ds:
            reasons = cache_mismatch_reasons(ds, args)
            if reasons:
                print(f"Cache exists but will be recomputed: {', '.join(reasons)}")
                return False

            print(f"Using cached WK spectra: {cache_file}")
            freqs = ds["frequency"].values
            ks = ds["zonal_wavenumber"].values
            plot_wk(
                ds["power_ens_mean"].values,
                freqs,
                ks,
                out_dir / f"{args.out_prefix}.png",
                f"RP ensemble mean {args.component} {args.var} WK spectrum",
                args.smooth_passes,
            )
            if args.plot_members:
                for member in ds.member.values:
                    member_name = str(member)
                    plot_wk(
                        ds["power"].sel(member=member_name).values,
                        freqs,
                        ks,
                        out_dir / f"{args.out_prefix}_{member_name}.png",
                        f"{member_name} {args.component} {args.var} WK spectrum",
                        args.smooth_passes,
                    )
            print(f"  Plot:   {out_dir / (args.out_prefix + '.png')}")
            return True
    except Exception as exc:
        print(f"Cache exists but could not be used: {exc}")
        return False


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    nc_file = out_dir / f"{args.out_prefix}.nc4"
    png_file = out_dir / f"{args.out_prefix}.png"

    if plot_from_cache(nc_file, out_dir, args):
        return

    print("Discovering RP member files...")
    files_by_member = {
        member: member_files(base_dir, member, args.collection, args.pattern)
        for member in args.members
    }
    for member, files in files_by_member.items():
        print(f"  {member}: {len(files)} files")
    missing = [member for member, files in files_by_member.items() if not files]
    if missing:
        raise FileNotFoundError(f"no files found for: {', '.join(missing)}")

    max_steps = None
    if args.max_days is not None:
        max_steps = int(round(args.max_days * 24.0 / args.dt_hours))
        print(f"Using first {max_steps} samples from each member.")
    elif not args.no_truncate_to_common:
        print("Counting time samples...")
        counts = {
            member: time_count(files, args.var, args.start_time, args.end_time)
            for member, files in files_by_member.items()
        }
        for member, count in counts.items():
            print(f"  {member}: {count} samples")
        empty_members = [member for member, count in counts.items() if count == 0]
        if empty_members:
            raise RuntimeError(f"no usable time samples for: {', '.join(empty_members)}")
        common_steps = min(counts.values())
        max_steps = common_steps
    if max_steps is not None:
        if max_steps < 2:
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

        if args.plot_members:
            plot_wk(
                power,
                freqs,
                ks,
                out_dir / f"{args.out_prefix}_{member}.png",
                f"{member} {args.component} {args.var} WK spectrum",
                args.smooth_passes,
            )

    member_power = np.stack(spectra, axis=0)
    power_ens = member_power.mean(axis=0)

    save_wk_dataset(nc_file, member_names, member_power, freqs_ref, ks_ref, max_steps, args)
    plot_wk(
        power_ens,
        freqs_ref,
        ks_ref,
        png_file,
        f"RP ensemble mean {args.component} {args.var} WK spectrum",
        args.smooth_passes,
    )

    print("\nDone.")
    print(f"  NetCDF: {nc_file}")
    print(f"  Plot:   {png_file}")


if __name__ == "__main__":
    main()
