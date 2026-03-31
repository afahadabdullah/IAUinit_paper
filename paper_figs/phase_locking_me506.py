"""
phase_locking_me506.py

Quantify phase locking between the ME506 OLR composite and spike composite saved
from OLR_composite.ipynb, and save a figure plus the derived metrics.

Expected input variables in the NetCDF file:
- wave(time, lon) or olr_506(time, lon) or olr_composite(time, lon)
- sp(time, lon) or spike_506(time, lon) or spike_composite(time, lon)
- lag_days(time) [optional]

Outputs:
- phase_locking_me506.png
- phase_locking_me506_metrics.nc
- phase_locking_me506_summary.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D


DEFAULT_INPUT = Path("phase_locking_data/phase_locking_ME506.nc")
DEFAULT_PLOT = Path("phase_locking_data/phase_locking_me506.png")
DEFAULT_METRICS = Path("phase_locking_data/phase_locking_me506_metrics.nc")
DEFAULT_SUMMARY = Path("phase_locking_data/phase_locking_me506_summary.txt")

OLR_CANDIDATES = ("wave", "olr_506", "olr_composite")
SPIKE_CANDIDATES = ("sp", "spike_506", "spike_composite")

DEFAULT_OLR_THRESHOLD = -5.0
ENVELOPE_HALF_WIDTH_DEG = 10.0
SPIKE_THRESHOLD = 0.25
MIN_SPIKE_SUM = 1.0e-8
MIN_OLR_WEIGHT_SUM = 1.0e-8
LAG_WINDOW_DAYS = 10.0
DEFAULT_N_PERM = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantify ME506 phase locking from saved OLR/spike composites."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input NetCDF path containing OLR and spike composites.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=DEFAULT_PLOT,
        help="Output PNG path for the phase-locking figure.",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS,
        help="Output NetCDF path for derived phase-locking metrics.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Output text path for a short numerical summary.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="Composite",
        help="Short label to use in figure and summary titles.",
    )
    parser.add_argument(
        "--olr-threshold",
        type=float,
        default=DEFAULT_OLR_THRESHOLD,
        help="Active-envelope threshold for OLR anomalies (W m^-2).",
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=DEFAULT_N_PERM,
        help="Number of circular-shift permutations for significance testing.",
    )
    return parser.parse_args()


def find_var(ds: xr.Dataset, candidates: tuple[str, ...]) -> xr.DataArray:
    for name in candidates:
        if name in ds:
            return ds[name]
    raise KeyError(f"None of the candidate variables were found: {candidates}")


def get_lag0_time(spike: xr.DataArray, ds: xr.Dataset) -> pd.Timestamp:
    if "lag0_time" in ds.attrs:
        return pd.Timestamp(ds.attrs["lag0_time"])

    if np.issubdtype(spike["time"].dtype, np.datetime64):
        sp_120w = spike.sel(lon=240, method="nearest")
        lag0_idx = int(sp_120w.argmax("time").item())
        return pd.Timestamp(sp_120w.time.isel(time=lag0_idx).values)

    time_vals = spike["time"].values
    return pd.Timestamp(time_vals[len(time_vals) // 2])


def normalize_lag_days(ds: xr.Dataset, spike: xr.DataArray) -> xr.DataArray:
    if "lag_days" in ds:
        lag = ds["lag_days"]
        if np.issubdtype(lag.dtype, np.datetime64):
            lag0_time = get_lag0_time(spike, ds)
            values = (
                (pd.to_datetime(lag.values) - lag0_time) / np.timedelta64(1, "D")
            ).astype(float)
        elif np.issubdtype(lag.dtype, np.timedelta64):
            values = (lag.values / np.timedelta64(1, "D")).astype(float)
        else:
            values = lag.values.astype(float)
        return xr.DataArray(values, dims=("time",), coords={"time": spike["time"]}, name="lag_days")

    if np.issubdtype(spike["time"].dtype, np.datetime64):
        lag0_time = get_lag0_time(spike, ds)
        values = (
            (pd.to_datetime(spike["time"].values) - lag0_time) / np.timedelta64(1, "D")
        ).astype(float)
        return xr.DataArray(values, dims=("time",), coords={"time": spike["time"]}, name="lag_days")

    values = np.arange(spike.sizes["time"], dtype=float)
    return xr.DataArray(values, dims=("time",), coords={"time": spike["time"]}, name="lag_days")


def circular_weighted_center(lon_vals: np.ndarray, weights: np.ndarray) -> float:
    """Return weighted center in degrees for 0-360 longitude arrays."""
    radians = np.deg2rad(lon_vals)
    z = np.sum(weights * np.exp(1j * radians))
    if np.abs(z) < MIN_OLR_WEIGHT_SUM:
        return np.nan
    center = np.rad2deg(np.angle(z)) % 360.0
    return float(center)


def signed_lon_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return signed longitude difference a-b in [-180, 180]."""
    return (a - b + 180.0) % 360.0 - 180.0


def compute_correlation(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan
    x_valid = x[valid]
    y_valid = y[valid]
    if np.allclose(x_valid.std(), 0.0) or np.allclose(y_valid.std(), 0.0):
        return np.nan
    return float(np.corrcoef(x_valid, y_valid)[0, 1])


def nearest_distance_to_active(lon: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    active_lons = lon[active_mask]
    if active_lons.size == 0:
        return np.full(lon.shape, np.nan, dtype=float)

    lon_2d = lon[:, None]
    active_2d = active_lons[None, :]
    return np.min(np.abs(signed_lon_difference(lon_2d, active_2d)), axis=1)


def row_metrics(
    olr_row: np.ndarray,
    spike_row: np.ndarray,
    lon: np.ndarray,
    olr_threshold: float,
) -> dict[str, float]:
    spike_weights = np.where(spike_row >= SPIKE_THRESHOLD, spike_row, 0.0)
    if np.nansum(spike_weights) <= MIN_SPIKE_SUM:
        spike_weights = np.clip(spike_row, a_min=0.0, a_max=None)

    out = {
        "spike_fraction_in_active_envelope": np.nan,
        "spike_fraction_within_buffer": np.nan,
        "active_area_fraction": np.nan,
        "buffer_area_fraction": np.nan,
        "active_enrichment": np.nan,
        "buffer_enrichment": np.nan,
        "mean_nearest_distance_deg": np.nan,
        "spatial_corr": compute_correlation(spike_row, -olr_row),
        "has_active_envelope": 0.0,
    }

    active_mask = olr_row <= olr_threshold
    if not active_mask.any():
        return out

    out["has_active_envelope"] = 1.0
    nearest_distance = nearest_distance_to_active(lon, active_mask)
    buffer_mask = nearest_distance <= ENVELOPE_HALF_WIDTH_DEG

    out["active_area_fraction"] = active_mask.mean()
    out["buffer_area_fraction"] = buffer_mask.mean()

    total_spike = np.nansum(spike_weights)
    if total_spike <= MIN_SPIKE_SUM:
        return out

    out["spike_fraction_in_active_envelope"] = np.nansum(spike_weights[active_mask]) / total_spike
    out["spike_fraction_within_buffer"] = np.nansum(spike_weights[buffer_mask]) / total_spike
    out["mean_nearest_distance_deg"] = np.nansum(spike_weights * nearest_distance) / total_spike

    if out["active_area_fraction"] > 0:
        out["active_enrichment"] = (
            out["spike_fraction_in_active_envelope"] / out["active_area_fraction"]
        )
    if out["buffer_area_fraction"] > 0:
        out["buffer_enrichment"] = (
            out["spike_fraction_within_buffer"] / out["buffer_area_fraction"]
        )

    return out


def build_metrics(
    olr: xr.DataArray,
    spike: xr.DataArray,
    lag_days: xr.DataArray,
    olr_threshold: float,
) -> xr.Dataset:
    lon = olr["lon"].values.astype(float)
    ntime = olr.sizes["time"]

    spike_fraction_in_active = np.full(ntime, np.nan, dtype=float)
    spike_fraction_within_buffer = np.full(ntime, np.nan, dtype=float)
    active_area_fraction = np.full(ntime, np.nan, dtype=float)
    buffer_area_fraction = np.full(ntime, np.nan, dtype=float)
    active_enrichment = np.full(ntime, np.nan, dtype=float)
    buffer_enrichment = np.full(ntime, np.nan, dtype=float)
    mean_nearest_distance = np.full(ntime, np.nan, dtype=float)
    spatial_corr = np.full(ntime, np.nan, dtype=float)
    has_active_envelope = np.zeros(ntime, dtype=int)

    for i in range(ntime):
        olr_row = olr.isel(time=i).values.astype(float)
        spike_row = spike.isel(time=i).values.astype(float)
        row = row_metrics(olr_row, spike_row, lon, olr_threshold)

        spike_fraction_in_active[i] = row["spike_fraction_in_active_envelope"]
        spike_fraction_within_buffer[i] = row["spike_fraction_within_buffer"]
        active_area_fraction[i] = row["active_area_fraction"]
        buffer_area_fraction[i] = row["buffer_area_fraction"]
        active_enrichment[i] = row["active_enrichment"]
        buffer_enrichment[i] = row["buffer_enrichment"]
        mean_nearest_distance[i] = row["mean_nearest_distance_deg"]
        spatial_corr[i] = row["spatial_corr"]
        has_active_envelope[i] = int(row["has_active_envelope"])

    return xr.Dataset(
        data_vars={
            "spike_fraction_in_active_envelope": ("time", spike_fraction_in_active),
            "spike_fraction_within_buffer": ("time", spike_fraction_within_buffer),
            "active_area_fraction": ("time", active_area_fraction),
            "buffer_area_fraction": ("time", buffer_area_fraction),
            "active_enrichment": ("time", active_enrichment),
            "buffer_enrichment": ("time", buffer_enrichment),
            "mean_nearest_distance_deg": ("time", mean_nearest_distance),
            "spatial_corr": ("time", spatial_corr),
            "has_active_envelope": ("time", has_active_envelope),
            "lag_days": lag_days,
        },
        coords={"time": olr["time"]},
        attrs={
            "negative_olr_threshold": olr_threshold,
            "envelope_half_width_deg": ENVELOPE_HALF_WIDTH_DEG,
            "spike_threshold": SPIKE_THRESHOLD,
            "lag_window_days": LAG_WINDOW_DAYS,
            "description": "Derived metrics for phase locking between OLR and spikes.",
        },
    )


def compute_significance(
    olr: xr.DataArray,
    spike: xr.DataArray,
    lag_days: xr.DataArray,
    olr_threshold: float,
    n_perm: int,
) -> xr.Dataset:
    lon = olr["lon"].values.astype(float)
    lag = lag_days.values.astype(float)
    ntime = olr.sizes["time"]
    rng = np.random.default_rng(42)

    p_active_fraction = np.full(ntime, np.nan, dtype=float)
    p_mean_distance = np.full(ntime, np.nan, dtype=float)

    analysis_mask = (
        np.isfinite(lag)
        & (np.abs(lag) >= 1.0)
        & (np.abs(lag) <= LAG_WINDOW_DAYS)
    )

    observed_active_window = []
    observed_distance_window = []
    null_active_window = []
    null_distance_window = []

    for i in range(ntime):
        if not analysis_mask[i]:
            continue

        olr_row = olr.isel(time=i).values.astype(float)
        spike_row = spike.isel(time=i).values.astype(float)
        obs = row_metrics(olr_row, spike_row, lon, olr_threshold)

        if not np.isfinite(obs["spike_fraction_in_active_envelope"]) or not np.isfinite(obs["mean_nearest_distance_deg"]):
            continue

        observed_active_window.append(obs["spike_fraction_in_active_envelope"])
        observed_distance_window.append(obs["mean_nearest_distance_deg"])

        null_active = []
        null_distance = []
        for _ in range(n_perm):
            shift = int(rng.integers(0, lon.size))
            shifted_spike = np.roll(spike_row, shift)
            perm = row_metrics(olr_row, shifted_spike, lon, olr_threshold)
            if np.isfinite(perm["spike_fraction_in_active_envelope"]):
                null_active.append(perm["spike_fraction_in_active_envelope"])
            if np.isfinite(perm["mean_nearest_distance_deg"]):
                null_distance.append(perm["mean_nearest_distance_deg"])

        if null_active:
            null_active = np.asarray(null_active, dtype=float)
            p_active_fraction[i] = (
                np.sum(null_active >= obs["spike_fraction_in_active_envelope"]) + 1.0
            ) / (null_active.size + 1.0)
            null_active_window.append(null_active)

        if null_distance:
            null_distance = np.asarray(null_distance, dtype=float)
            p_mean_distance[i] = (
                np.sum(null_distance <= obs["mean_nearest_distance_deg"]) + 1.0
            ) / (null_distance.size + 1.0)
            null_distance_window.append(null_distance)

    global_p_active_fraction = np.nan
    global_p_mean_distance = np.nan

    if observed_active_window and null_active_window:
        observed_mean_active = float(np.mean(observed_active_window))
        n_common = min(arr.size for arr in null_active_window)
        null_mean_active = np.array(
            [np.mean([arr[k] for arr in null_active_window]) for k in range(n_common)],
            dtype=float,
        )
        global_p_active_fraction = (
            np.sum(null_mean_active >= observed_mean_active) + 1.0
        ) / (null_mean_active.size + 1.0)

    if observed_distance_window and null_distance_window:
        observed_mean_distance = float(np.mean(observed_distance_window))
        n_common = min(arr.size for arr in null_distance_window)
        null_mean_distance = np.array(
            [np.mean([arr[k] for arr in null_distance_window]) for k in range(n_common)],
            dtype=float,
        )
        global_p_mean_distance = (
            np.sum(null_mean_distance <= observed_mean_distance) + 1.0
        ) / (null_mean_distance.size + 1.0)

    return xr.Dataset(
        data_vars={
            "p_active_fraction": ("time", p_active_fraction),
            "p_mean_distance": ("time", p_mean_distance),
        },
        coords={"time": olr["time"]},
        attrs={
            "n_perm": n_perm,
            "global_p_active_fraction": global_p_active_fraction,
            "global_p_mean_distance": global_p_mean_distance,
            "significance_method": "One-sided circular-shift permutation test along longitude at each lag.",
        },
    )


def format_lon_labels(ax: plt.Axes, xticks: np.ndarray) -> None:
    labels = []
    for lon in xticks:
        if lon <= 180:
            labels.append(f"{lon:.0f}\N{DEGREE SIGN}E")
        else:
            labels.append(f"{360 - lon:.0f}\N{DEGREE SIGN}W")
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels)


def make_plot(
    olr: xr.DataArray,
    spike: xr.DataArray,
    lag_days: xr.DataArray,
    metrics: xr.Dataset,
    output_path: Path,
    label: str,
    olr_threshold: float,
) -> None:
    full_lag = lag_days.values.astype(float)
    plot_mask = np.isfinite(full_lag) & (np.abs(full_lag) <= LAG_WINDOW_DAYS)
    plot_times = lag_days["time"].values[plot_mask]

    lag_win = lag_days.sel(time=plot_times)
    metrics_win = metrics.sel(time=plot_times)

    lag = lag_win.values.astype(float)
    fig = plt.figure(figsize=(11.5, 8.5))

    ax1 = fig.add_subplot(2, 1, 1)
    levels = np.array([-50, -40, -30, -20, -10, -0.5, 0.5, 10, 20, 30, 40, 50])
    cf = olr.plot.contourf(
        ax=ax1,
        levels=levels,
        extend="both",
        alpha=0.85,
        cmap="RdBu_r",
        add_colorbar=False,
    )
    spike.plot.contour(
        ax=ax1,
        colors="black",
        levels=np.array([SPIKE_THRESHOLD]),
        alpha=0.5,
        add_colorbar=False,
    )
    olr.plot.contour(
        ax=ax1,
        colors="gold",
        levels=np.array([olr_threshold]),
        linewidths=2.0,
        linestyles="--",
        alpha=0.9,
        add_colorbar=False,
    )
    format_lon_labels(ax1, np.arange(120, 300, 30))

    time_values = olr["time"].values
    tick_offsets = np.arange(0, 28, 4)
    tick_times = pd.to_datetime(time_values[0]) + pd.to_timedelta(tick_offsets, unit="D")
    valid_ticks = tick_times <= pd.to_datetime(time_values[-1])
    ax1.set_yticks(tick_times[valid_ticks])
    ax1.set_yticklabels((tick_offsets[valid_ticks] - 10).astype(int), fontsize=12)
    ax1.set_ylabel("Lag/Lead days", fontsize=14)

    x_mark = 240.0
    y_mark = pd.to_datetime(time_values[0]) + pd.Timedelta(days=11)
    ax1.plot(
        x_mark,
        y_mark,
        marker="o",
        markersize=40,
        color="yellow",
        markerfacecolor="none",
        markeredgewidth=5,
    )
    ax1.set_title(
        "Composite mean OLR anomalies during Eastern Pacific Spikes",
        fontsize=13,
        fontweight="bold",
    )
    ax1.legend(
        handles=[
            Line2D([0], [0], color="gold", lw=2.0, ls="--", label=f"Active OLR envelope ({olr_threshold:.0f} W m$^{{-2}}$)"),
            Line2D([0], [0], color="black", lw=1.5, label="Spike contour (0.25)"),
        ],
        loc="upper left",
        frameon=True,
    )
    fig.colorbar(cf, ax=ax1, pad=0.01, label="OLR anomaly")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.axvline(0.0, color="0.6", lw=1.0, ls="--")
    ax2.plot(
        lag,
        metrics_win["spike_fraction_in_active_envelope"],
        color="tab:red",
        lw=2.5,
        label="Spike fraction inside active envelope",
    )
    ax2.set_ylabel("Spike mass fraction")
    ax2.set_xlabel("Lag / lead days")
    ax2.grid(alpha=0.3)

    ax2b = ax2.twinx()
    ax2b.plot(
        lag,
        metrics_win["mean_nearest_distance_deg"],
        color="tab:purple",
        lw=2.5,
        label="Mean nearest distance",
    )
    ax2b.set_ylabel("Mean nearest distance (deg)")

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True)
    ax2.set_title(
        f"Spike-envelope overlap and distance within +/-{int(LAG_WINDOW_DAYS)} days",
        fontweight="bold",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary(metrics: xr.Dataset, significance: xr.Dataset, output_path: Path, label: str) -> None:
    lag = metrics["lag_days"].values.astype(float)
    active_enrich = metrics["active_enrichment"].values.astype(float)
    buffer_enrich = metrics["buffer_enrichment"].values.astype(float)
    active_frac = metrics["spike_fraction_in_active_envelope"].values.astype(float)
    buffer_frac = metrics["spike_fraction_within_buffer"].values.astype(float)
    distance = metrics["mean_nearest_distance_deg"].values.astype(float)
    corr = metrics["spatial_corr"].values.astype(float)
    has_active = metrics["has_active_envelope"].values.astype(int)
    p_active = significance["p_active_fraction"].values.astype(float)
    p_distance = significance["p_mean_distance"].values.astype(float)

    analysis_mask = (
        np.isfinite(lag)
        & (has_active == 1)
        & (np.abs(lag) >= 1.0)
        & (np.abs(lag) <= LAG_WINDOW_DAYS)
    )

    def safe_mean(values: np.ndarray, mask: np.ndarray) -> float:
        if not mask.any():
            return float("nan")
        return float(np.nanmean(values[mask]))

    def safe_median(values: np.ndarray, mask: np.ndarray) -> float:
        if not mask.any():
            return float("nan")
        return float(np.nanmedian(values[mask]))

    mean_active_fraction = safe_mean(active_frac, analysis_mask)
    mean_buffer_fraction = safe_mean(buffer_frac, analysis_mask)
    mean_active_enrichment = safe_mean(active_enrich, analysis_mask)
    mean_buffer_enrichment = safe_mean(buffer_enrich, analysis_mask)
    mean_distance = safe_mean(distance, analysis_mask)
    median_distance = safe_median(distance, analysis_mask)
    mean_corr = safe_mean(corr, analysis_mask)
    close_fraction = (
        float(np.nanmean((distance[analysis_mask] <= ENVELOPE_HALF_WIDTH_DEG).astype(float)))
        if analysis_mask.any()
        else float("nan")
    )
    sig_active_count = (
        int(np.sum((p_active[analysis_mask] < 0.05) & np.isfinite(p_active[analysis_mask])))
        if analysis_mask.any()
        else 0
    )
    sig_distance_count = (
        int(np.sum((p_distance[analysis_mask] < 0.05) & np.isfinite(p_distance[analysis_mask])))
        if analysis_mask.any()
        else 0
    )
    global_p_active = significance.attrs.get("global_p_active_fraction", np.nan)
    global_p_distance = significance.attrs.get("global_p_mean_distance", np.nan)
    n_perm = significance.attrs.get("n_perm", DEFAULT_N_PERM)

    summary = [
        f"{label} phase-locking summary",
        f"Analysis lags used: {int(analysis_mask.sum())}",
        f"Lag window analyzed: |lag| <= {LAG_WINDOW_DAYS:.0f} days, excluding lag 0",
        f"Active envelope threshold: OLR <= {metrics.attrs['negative_olr_threshold']:.0f} W m^-2",
        f"Mean spike fraction inside active envelope: {mean_active_fraction:.3f}",
        f"Mean spike fraction within +/-{ENVELOPE_HALF_WIDTH_DEG:.0f} deg of active envelope: {mean_buffer_fraction:.3f}",
        f"Mean active-envelope enrichment: {mean_active_enrichment:.2f}",
        f"Mean +/-{ENVELOPE_HALF_WIDTH_DEG:.0f} deg envelope enrichment: {mean_buffer_enrichment:.2f}",
        f"Mean spike-weighted nearest distance to active envelope: {mean_distance:.2f} deg",
        f"Median spike-weighted nearest distance to active envelope: {median_distance:.2f} deg",
        f"Fraction of analyzed lags with mean nearest distance <= {ENVELOPE_HALF_WIDTH_DEG:.0f} deg: {close_fraction:.3f}",
        f"Mean lon correlation corr(spike, -OLR): {mean_corr:.2f}",
        f"Global p-value for mean spike fraction inside active envelope: {global_p_active:.4f}",
        f"Global p-value for mean nearest distance: {global_p_distance:.4f}",
        f"Lags with p < 0.05 for spike fraction inside active envelope: {sig_active_count}/{int(analysis_mask.sum())}",
        f"Lags with p < 0.05 for mean nearest distance: {sig_distance_count}/{int(analysis_mask.sum())}",
        "",
        "Metric definitions:",
        "- Active envelope = longitudes where OLR is below the negative threshold.",
        "- Spike fraction inside active envelope = sum of spike mass at longitudes with OLR <= threshold, divided by total spike mass at that lag.",
        f"- Mean nearest distance = spike-weighted mean longitude distance to the nearest active-envelope longitude.",
        "- Active-envelope enrichment = [spike fraction inside active envelope] / [fraction of longitudes inside the active envelope].",
        "- Positive spatial correlation means spike maxima align with negative OLR anomalies across longitude.",
        "",
        "Statistical significance:",
        f"- Significance uses a one-sided circular-shift permutation test along longitude with {n_perm} permutations at each lag.",
        "- For spike fraction, the p-value is the fraction of random longitude shifts that produce overlap at least as large as observed.",
        "- For mean nearest distance, the p-value is the fraction of random longitude shifts that produce a distance at least as small as observed.",
        "- Smaller p-values indicate stronger phase alignment than expected from random longitudinal placement of the spike pattern.",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(summary) + "\n")


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}\n"
            "Create it first from OLR_composite.ipynb."
        )

    ds = xr.open_dataset(args.input)
    olr = find_var(ds, OLR_CANDIDATES).transpose("time", "lon")
    spike = find_var(ds, SPIKE_CANDIDATES).transpose("time", "lon")
    lag_days = normalize_lag_days(ds, spike)

    metrics = build_metrics(
        olr=olr,
        spike=spike,
        lag_days=lag_days,
        olr_threshold=args.olr_threshold,
    )
    significance = compute_significance(
        olr=olr,
        spike=spike,
        lag_days=lag_days,
        olr_threshold=args.olr_threshold,
        n_perm=args.n_perm,
    )

    metrics_out = metrics.copy()
    for name, data_array in significance.data_vars.items():
        metrics_out[name] = data_array
    metrics_out.attrs.update(significance.attrs)

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.to_netcdf(args.metrics)

    write_summary(metrics, significance, args.summary, args.label)
    make_plot(
        olr=olr,
        spike=spike,
        lag_days=lag_days,
        metrics=metrics,
        output_path=args.plot,
        label=args.label,
        olr_threshold=args.olr_threshold,
    )

    print(f"Saved plot: {args.plot}")
    print(f"Saved metrics: {args.metrics}")
    print(f"Saved summary: {args.summary}")


if __name__ == "__main__":
    main()
