#!/usr/bin/env python3
"""Lag-lead correlation between cached precipitation spikes and MC.

This script is intentionally cache-only: it reads the point moisture-budget
NetCDF files written by mse_budget2.py and does not reopen the large model
history files.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

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
INITIAL_EVENT_HOURS = 72.0
SIGNIFICANCE_LEVEL = 0.05
SECONDS_PER_DAY = 86400.0

REGIONS = [
    {"key": "wtp", "title": "Western Tropical Pacific"},
    {"key": "ctp", "title": "Central Tropical Pacific"},
    {"key": "etp", "title": "Eastern Tropical Pacific"},
    {"key": "itf", "title": "Indonesian Throughflow"},
    {"key": "tio", "title": "Tropical Indian Ocean"},
    {"key": "tao", "title": "Tropical Atlantic Ocean"},
]

EXPERIMENTS = ("Reanalysis-IC",)
EVENT_GROUPS = (
    ("all", "All spike events", "black", "o", "solid"),
    ("initial", "Initial events <= 3 days", "tab:blue", "s", "dashed"),
    ("later", "Events > 3 days", "tab:orange", "^", "dashdot"),
)
CACHE_DIR = Path(__file__).with_name("cache_mse_budget2_direct_mc_points_sync")
LEGACY_WTP_CACHE_DIR = Path(__file__).with_name("cache_mse_budget2_direct_mc_wtp_point_sync")
CSV_PATH = Path(__file__).with_name("mse_budget2_spike_lag_lead_correlation.csv")
FIGURE_PATH = Path(__file__).with_name("mse_budget2_spike_lag_lead_correlation.png")
SIGN_CONVENTION = "corr(P(t), MC(t + lag)); negative lag means MC leads precipitation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute lag-lead correlations between cached precipitation and MC "
            "using only precipitation-spike event windows."
        )
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=CACHE_DIR,
        help="Directory containing <experiment>_<region>_direct_mc.nc cache files.",
    )
    parser.add_argument(
        "--legacy-wtp-cache-dir",
        type=Path,
        default=LEGACY_WTP_CACHE_DIR,
        help="Fallback WTP cache directory used only if WTP is missing from --cache-dir.",
    )
    parser.add_argument(
        "--regions",
        default="all",
        help="Comma-separated region keys to analyze, or 'all'.",
    )
    parser.add_argument("--lag-min-hours", type=float, default=-6.0)
    parser.add_argument("--lag-max-hours", type=float, default=6.0)
    parser.add_argument("--lag-step-hours", type=float, default=6.0)
    parser.add_argument(
        "--lag-hours",
        default=None,
        help="Optional comma-separated lag list. Overrides min/max/step.",
    )
    parser.add_argument("--spike-quantile", type=float, default=SPIKE_QUANTILE)
    parser.add_argument("--event-window-hours", type=float, default=EVENT_WINDOW_HOURS)
    parser.add_argument(
        "--initial-event-hours",
        type=float,
        default=INITIAL_EVENT_HOURS,
        help="Spike peaks within this many hours of the simulation start are grouped as initial events.",
    )
    parser.add_argument("--precip-smooth-hours", type=float, default=PRECIP_SMOOTH_HOURS)
    parser.add_argument(
        "--correlation-smooth-hours",
        type=float,
        default=0.0,
        help="Optional smoothing applied to P and MC before collecting correlation samples.",
    )
    parser.add_argument(
        "--mc-time-mean-hours",
        type=float,
        default=0.0,
        help="Optional centered time-mean approximation applied to cached instantaneous MC before correlation.",
    )
    parser.add_argument("--output-csv", type=Path, default=CSV_PATH)
    parser.add_argument("--output-figure", type=Path, default=FIGURE_PATH)
    parser.add_argument("--no-plot", action="store_true", help="Skip writing the lag-correlation plot.")
    return parser.parse_args()


def parse_regions(value: str) -> list[dict[str, str]]:
    if value.lower() == "all":
        return REGIONS
    wanted = {item.strip().lower() for item in value.split(",") if item.strip()}
    regions = [region for region in REGIONS if region["key"] in wanted]
    missing = wanted.difference(region["key"] for region in regions)
    if missing:
        raise ValueError(f"Unknown region key(s): {sorted(missing)}")
    return regions


def parse_lags(args: argparse.Namespace) -> np.ndarray:
    if args.lag_hours:
        return np.asarray([float(item.strip()) for item in args.lag_hours.split(",") if item.strip()], dtype=float)
    if args.lag_step_hours <= 0.0:
        raise ValueError("--lag-step-hours must be positive.")
    return np.arange(
        args.lag_min_hours,
        args.lag_max_hours + 0.5 * args.lag_step_hours,
        args.lag_step_hours,
        dtype=float,
    )


def cache_candidates(cache_dir: Path, legacy_wtp_cache_dir: Path, name: str, region: dict[str, str]) -> list[Path]:
    candidates = [cache_dir / f"{name}_{region['key']}_direct_mc.nc"]
    if region["key"] == "wtp":
        candidates.append(legacy_wtp_cache_dir / f"{name}_{region['key']}_direct_mc.nc")
    return candidates


def centered_time_mean_from_instantaneous(da: xr.DataArray, window_hours: float) -> xr.DataArray:
    if window_hours <= 0.0 or da.sizes.get("time", 0) <= 1:
        return da.copy()

    times = da.time.values.astype("datetime64[s]").astype(np.float64)
    values = np.asarray(da.values, dtype=float)
    finite = np.isfinite(times) & np.isfinite(values)
    if finite.sum() <= 1:
        return da.copy()

    source_times = times[finite]
    source_values = values[finite]
    order = np.argsort(source_times)
    source_times = source_times[order]
    source_values = source_values[order]

    half_window_seconds = 0.5 * window_hours * 3600.0
    averaged = np.full(values.shape, np.nan, dtype=float)
    for idx, time_value in enumerate(times):
        if not np.isfinite(time_value):
            continue
        start = max(time_value - half_window_seconds, source_times[0])
        end = min(time_value + half_window_seconds, source_times[-1])
        if end <= start:
            averaged[idx] = np.interp(time_value, source_times, source_values)
            continue

        inner = source_times[(source_times > start) & (source_times < end)]
        eval_times = np.unique(np.concatenate(([start], inner, [end])))
        eval_values = np.interp(eval_times, source_times, source_values)
        averaged[idx] = float(np.trapz(eval_values, eval_times) / (end - start))

    return xr.DataArray(averaged, dims=da.dims, coords=da.coords, attrs=dict(da.attrs), name=da.name)


def apply_mc_time_mean(ds: xr.Dataset, window_hours: float) -> xr.Dataset:
    if window_hours <= 0.0:
        return ds
    out = ds.copy()
    source = out["MCDirectInstantaneous"] if "MCDirectInstantaneous" in out else out["MCDirect"]
    if "MCDirectInstantaneous" not in out:
        out["MCDirectInstantaneous"] = source.copy()
    out["MCDirect"] = centered_time_mean_from_instantaneous(source, window_hours)
    out.attrs["mc_time_mean_applied_hours"] = float(window_hours)
    return out


def load_cached_series(
    cache_dir: Path,
    legacy_wtp_cache_dir: Path,
    name: str,
    region: dict[str, str],
    mc_time_mean_hours: float,
) -> tuple[xr.Dataset | None, Path | None]:
    for path in cache_candidates(cache_dir, legacy_wtp_cache_dir, name, region):
        if not path.exists():
            continue
        ds = xr.open_dataset(path).load()
        ds = ds.sel(time=slice(ANALYSIS_START, ANALYSIS_END))
        missing = [var for var in ("Precip", "MCDirect") if var not in ds]
        if missing:
            raise ValueError(f"{path} is missing required variable(s): {missing}")
        return apply_mc_time_mean(ds, mc_time_mean_hours), path
    return None, None


def smoothed_pair_dataset(ds: xr.Dataset, smooth_hours: float) -> xr.Dataset:
    steps = mb.smooth_steps_from_hours(ds, smooth_hours)
    if steps <= 1:
        return ds[["Precip", "MCDirect"]].copy()
    out = xr.Dataset(attrs=dict(ds.attrs))
    out["Precip"] = mb.rolling_event_mean(ds["Precip"], steps)
    out["MCDirect"] = mb.rolling_event_mean(ds["MCDirect"], steps)
    return out


def detect_spikes(series: xr.Dataset, args: argparse.Namespace) -> tuple[np.ndarray, float]:
    detect = smoothed_pair_dataset(series, args.precip_smooth_hours)
    reference_precip = detect["Precip"]
    dt_hours = mb.infer_time_step_hours(detect)
    min_sep_steps = max(1, int(round(MIN_PEAK_SEPARATION_HOURS / max(dt_hours, 1.0e-6))))
    return mb.detect_precip_spikes(
        reference_precip,
        quantile=args.spike_quantile,
        min_separation=min_sep_steps,
        max_spikes=999,
    )


def simulation_start_time(series: xr.Dataset) -> np.datetime64:
    for attr_name in ("analysis_start", "start_time"):
        value = series.attrs.get(attr_name)
        if value:
            return np.datetime64(value, "s")
    return series.time.values[0].astype("datetime64[s]")


def split_spikes_by_start_time(series: xr.Dataset, spike_indices: np.ndarray, initial_hours: float) -> dict[str, np.ndarray]:
    if len(spike_indices) == 0:
        empty = np.asarray([], dtype=int)
        return {"all": empty, "initial": empty, "later": empty}

    start_time = simulation_start_time(series)
    cutoff = start_time + np.timedelta64(int(round(initial_hours * 3600.0)), "s")
    peak_times = series.time.isel(time=spike_indices).values.astype("datetime64[s]")
    initial_mask = peak_times <= cutoff
    initial = spike_indices[initial_mask]
    later = spike_indices[~initial_mask]
    return {
        "all": spike_indices,
        "initial": initial,
        "later": later,
    }


def collect_lag_pairs(
    series: xr.Dataset,
    spike_indices: np.ndarray,
    lag_hours: float,
    event_window_hours: float,
    correlation_smooth_hours: float,
) -> tuple[np.ndarray, np.ndarray]:
    ds = smoothed_pair_dataset(series, correlation_smooth_hours)
    dt_hours = mb.infer_time_step_hours(ds)
    lag_steps_float = lag_hours / max(dt_hours, 1.0e-6)
    lag_steps = int(round(lag_steps_float))
    if not np.isclose(lag_steps_float, lag_steps, atol=0.01):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    half_steps = mb.half_window_steps_from_hours(ds, event_window_hours)
    precip = np.asarray(ds["Precip"].values, dtype=float) * SECONDS_PER_DAY
    mc = np.asarray(ds["MCDirect"].values, dtype=float) * SECONDS_PER_DAY

    p_samples = []
    mc_samples = []
    n_time = ds.sizes.get("time", 0)
    for peak_idx in spike_indices:
        i0 = max(int(peak_idx) - half_steps, 0)
        i1 = min(int(peak_idx) + half_steps, n_time - 1)
        for p_idx in range(i0, i1 + 1):
            mc_idx = p_idx + lag_steps
            if mc_idx < 0 or mc_idx >= n_time:
                continue
            p_value = precip[p_idx]
            mc_value = mc[mc_idx]
            if np.isfinite(p_value) and np.isfinite(mc_value):
                p_samples.append(float(p_value))
                mc_samples.append(float(mc_value))

    return np.asarray(p_samples, dtype=float), np.asarray(mc_samples, dtype=float)


def correlation_row(
    region: dict[str, str],
    experiment: str,
    event_group_key: str,
    event_group_label: str,
    lag_hours: float,
    spike_count: int,
    threshold: float,
    p_samples: np.ndarray,
    mc_samples: np.ndarray,
    args: argparse.Namespace,
    cache_file: Path | str,
) -> dict[str, object]:
    n_pairs = int(len(p_samples))
    if n_pairs >= 3 and np.nanstd(p_samples) > 0.0 and np.nanstd(mc_samples) > 0.0:
        pearson = stats.pearsonr(p_samples, mc_samples)
        r_value = float(pearson.statistic)
        p_value = float(pearson.pvalue)
    else:
        r_value = np.nan
        p_value = np.nan

    return {
        "region_key": region["key"],
        "region_name": region["title"],
        "experiment": experiment,
        "event_group": event_group_key,
        "event_group_label": event_group_label,
        "lag_hours": float(lag_hours),
        "sign_convention": SIGN_CONVENTION,
        "n_spikes": int(spike_count),
        "n_pairs": n_pairs,
        "pearson_r": r_value,
        "pvalue": p_value,
        "threshold_mm_day": float(threshold * SECONDS_PER_DAY),
        "precip_mean_mm_day": float(np.nanmean(p_samples)) if n_pairs else np.nan,
        "mc_mean_mm_day": float(np.nanmean(mc_samples)) if n_pairs else np.nan,
        "precip_std_mm_day": float(np.nanstd(p_samples)) if n_pairs else np.nan,
        "mc_std_mm_day": float(np.nanstd(mc_samples)) if n_pairs else np.nan,
        "event_window_hours": float(args.event_window_hours),
        "initial_event_hours": float(args.initial_event_hours),
        "precip_smooth_hours_for_detection": float(args.precip_smooth_hours),
        "correlation_smooth_hours": float(args.correlation_smooth_hours),
        "mc_time_mean_hours": float(args.mc_time_mean_hours),
        "mc_lag_sampling": "actual cached MC time steps only; no off-cadence interpolation",
        "cache_file": str(cache_file),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {path}")


def plot_rows(path: Path, rows: list[dict[str, object]], regions: list[dict[str, str]]) -> None:
    experiment = EXPERIMENTS[0]
    selected_by_group = {}
    for group_key, group_label, *_ in EVENT_GROUPS:
        selected = [
            row
            for row in rows
            if row["region_key"] == "all"
            and row["experiment"] == experiment
            and row["event_group"] == group_key
        ]
        if selected:
            selected_by_group[group_key] = sorted(selected, key=lambda row: float(row["lag_hours"]))

    if not selected_by_group:
        raise ValueError("No pooled all-event rows available to plot.")

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axhline(0, color="0.6", linewidth=0.8)
    ax.axvline(0, color="0.6", linewidth=0.8, linestyle="--")
    for group_key, group_label, color, marker, linestyle in EVENT_GROUPS:
        selected = selected_by_group.get(group_key)
        if not selected:
            continue
        lags = np.asarray([float(row["lag_hours"]) for row in selected], dtype=float)
        values = np.asarray([float(row["pearson_r"]) for row in selected], dtype=float)
        pvalues = np.asarray([float(row["pvalue"]) for row in selected], dtype=float)
        ax.plot(
            lags,
            values,
            color=color,
            linestyle=linestyle,
            linewidth=2.4,
            alpha=0.75,
            label=group_label,
        )
        finite = np.isfinite(lags) & np.isfinite(values)
        significant = finite & np.isfinite(pvalues) & (pvalues < SIGNIFICANCE_LEVEL)
        nonsignificant = finite & ~significant
        ax.scatter(
            lags[nonsignificant],
            values[nonsignificant],
            marker=marker,
            s=58,
            color=color,
            alpha=0.25,
            edgecolors="none",
            zorder=3,
        )
        ax.scatter(
            lags[significant],
            values[significant],
            marker=marker,
            s=110,
            color=color,
            alpha=1.0,
            edgecolors="black",
            linewidths=1.4,
            zorder=4,
        )

    ax.set_title("(C) P(t) vs MC (t + lag/lead) correlation", fontsize=16, fontweight="bold")
    ax.set_xlabel("Lag hours")
    ax.set_ylabel("Pearson r")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower center", frameon=True, framealpha=0.9, facecolor="white", edgecolor="0.8")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def main() -> None:
    args = parse_args()
    regions = parse_regions(args.regions)
    lags = parse_lags(args)

    print("Spike-window lag-lead correlation from cached moisture-budget data")
    print(f"  Cache directory : {args.cache_dir}")
    print(f"  Regions         : {', '.join(region['key'] for region in regions)}")
    print(f"  Lag convention  : {SIGN_CONVENTION}")
    print(f"  Lags            : {', '.join(f'{lag:g}' for lag in lags)} h")
    print(f"  Event window    : +/-{0.5 * args.event_window_hours:g} h around each spike")
    print(f"  Initial split   : peak time <= simulation start + {args.initial_event_hours:g} h")
    print(f"  MC time mean    : {args.mc_time_mean_hours:g} h (0 means cached MCDirect unchanged)")
    print("  MC lag sampling : actual cached MC time steps only; no off-cadence interpolation")

    rows = []
    pooled_samples: dict[tuple[str, str, float], tuple[list[float], list[float], int]] = {}

    for region in regions:
        print(f"\n=== {region['title']} ({region['key']}) ===")
        loaded: dict[str, xr.Dataset] = {}
        cache_files: dict[str, Path] = {}
        for experiment in EXPERIMENTS:
            ds, path = load_cached_series(
                args.cache_dir,
                args.legacy_wtp_cache_dir,
                experiment,
                region,
                args.mc_time_mean_hours,
            )
            if ds is None or path is None:
                print(f"  Missing cache for {experiment}; skipping region.")
                loaded = {}
                break
            loaded[experiment] = ds
            cache_files[experiment] = path
            dt_hours = mb.infer_time_step_hours(ds)
            print(
                f"  {experiment}: {ds.sizes.get('time', 0)} time steps, "
                f"dt~{dt_hours:g} h, cache={path}"
            )

        if not loaded:
            continue

        spike_indices, threshold = detect_spikes(loaded["Reanalysis-IC"], args)
        print(
            f"  Spikes detected : {len(spike_indices)} above q={args.spike_quantile:.2f} "
            f"(threshold={threshold * SECONDS_PER_DAY:.2f} mm day-1)"
        )
        if len(spike_indices) == 0:
            continue

        spike_groups = split_spikes_by_start_time(
            loaded["Reanalysis-IC"],
            spike_indices,
            args.initial_event_hours,
        )
        print(
            "  Spike split     : "
            f"all={len(spike_groups['all'])}, "
            f"initial<={args.initial_event_hours / 24.0:g}d={len(spike_groups['initial'])}, "
            f"later={len(spike_groups['later'])}"
        )

        for experiment in EXPERIMENTS:
            lag0_pairs_by_group = {}
            for group_key, *_ in EVENT_GROUPS:
                group_spikes = spike_groups[group_key]
                for lag in lags:
                    p_samples, mc_samples = collect_lag_pairs(
                        loaded[experiment],
                        group_spikes,
                        float(lag),
                        args.event_window_hours,
                        args.correlation_smooth_hours,
                    )
                    if np.isclose(lag, 0.0):
                        lag0_pairs_by_group[group_key] = len(p_samples)

                    key = (group_key, experiment, float(lag))
                    if key not in pooled_samples:
                        pooled_samples[key] = ([], [], 0)
                    pooled_p, pooled_mc, pooled_spikes = pooled_samples[key]
                    pooled_p.extend(p_samples.tolist())
                    pooled_mc.extend(mc_samples.tolist())
                    pooled_samples[key] = (pooled_p, pooled_mc, pooled_spikes + len(group_spikes))

            lag0_text = ", ".join(
                f"{group_key}={lag0_pairs_by_group.get(group_key, 0)}"
                for group_key, *_ in EVENT_GROUPS
            )
            print(
                f"  {experiment}: spike events={len(spike_indices)}, "
                f"lag-0 paired samples ({lag0_text})"
            )

    group_labels = {group_key: group_label for group_key, group_label, *_ in EVENT_GROUPS}
    for (group_key, experiment, lag), (p_values, mc_values, pooled_spikes) in sorted(pooled_samples.items()):
        pooled_region = {"key": "all", "title": "All regions"}
        rows.append(
            correlation_row(
                pooled_region,
                experiment,
                group_key,
                group_labels[group_key],
                lag,
                pooled_spikes,
                np.nan,
                np.asarray(p_values, dtype=float),
                np.asarray(mc_values, dtype=float),
                args,
                "pooled caches",
            )
        )

    if not rows:
        raise SystemExit("No cache-backed lag-lead rows were produced.")

    write_csv(args.output_csv, rows)
    if not args.no_plot:
        plot_rows(args.output_figure, rows, regions)

    print("\nLag-0 pooled sample counts")
    for experiment in EXPERIMENTS:
        for group_key, group_label, *_ in EVENT_GROUPS:
            matches = [
                row
                for row in rows
                if row["region_key"] == "all"
                and row["experiment"] == experiment
                and row["event_group"] == group_key
                and np.isclose(row["lag_hours"], 0.0)
            ]
            if matches:
                row = matches[0]
                print(f"  {experiment} {group_label}: spikes={row['n_spikes']}, paired samples={row['n_pairs']}")


if __name__ == "__main__":
    main()
