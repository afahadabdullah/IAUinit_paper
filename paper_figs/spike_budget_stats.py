import csv
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import xarray as xr

import mse_budget as mb


# ==============================================================================
# Configuration
# ==============================================================================
ANALYSIS_START = "2005-05-01"
ANALYSIS_END = "2005-06-30"
SPIKE_QUANTILE = 0.95
MIN_PEAK_SEPARATION_HOURS = 24.0
EVENT_WINDOW_HOURS = 24.0
EVENT_WINDOW_LABEL = f"{EVENT_WINDOW_HOURS:g}-h spike window"
PRECIP_SMOOTH_HOURS = 6.0
BUDGET_SMOOTH_HOURS = 0.0
MAX_SPIKES_PER_REGION = 999
BOOTSTRAP_SAMPLES = 20000
RNG_SEED = 42
ENSEMBLE_EQUIVALENT_FACTOR = 10
ROBUST_TRIM_FRACTION = 0.20
SECONDS_PER_DAY = 86400.0

CACHE_DIR = Path(__file__).with_name("cache_multi_moisture_v1")
EVENT_TABLE_PATH = Path(__file__).with_name("multi_spike_moisture_budget_events.csv")
SUMMARY_PATH = Path(__file__).with_name("multi_spike_moisture_budget_summary.csv")
REGION_SUMMARY_PATH = Path(__file__).with_name("multi_spike_moisture_budget_region_summary.csv")
FIGURE_PATH = Path(__file__).with_name("multi_spike_budget_stats.png")

REGIONS = [
    {"key": "wtp", "title": "Western Tropical Pacific", "lon": (143.0, 143.0), "lat": (-1.0, -1.0)},
    {"key": "ctp", "title": "Central Tropical Pacific", "lon": (-145.0, -145.0), "lat": (8.0, 8.0)},
    {"key": "etp", "title": "Eastern Tropical Pacific", "lon": (-127.0, -127.0), "lat": (7.0, 7.0)},
    {"key": "itf", "title": "Indonesian Throughflow", "lon": (114.0, 115.0), "lat": (5.0, 6.0)},
    {"key": "tio", "title": "Tropical Indian Ocean", "lon": (89.0, 92.0), "lat": (7.0, 8.0)},
    {"key": "tao", "title": "Tropical Atlantic Ocean", "lon": (-39.0, -38.0), "lat": (7.0, 8.0)},
]
PACIFIC_REGION_KEYS = {"wtp", "ctp", "etp"}

PROG_REQUIRED_VARS = ("T", "QV")
PROG_OPTIONAL_VARS = mb.PROG_OPTIONAL_VARS
SURF_REQUIRED_VARS = ("PS", "LHFX")
SURF_OPTIONAL_VARS = ("PRECTOT", "TPREC", "precip")

COMPONENTS = [
    ("precip_mm", r"$\int P\,dt$"),
    ("evap_mm", r"$\int E\,dt$"),
    ("conv_mm", r"$\int MC\,dt$"),
    ("storage_mm", r"$-\Delta W$"),
]


# ==============================================================================
# Helpers
# ==============================================================================
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


def make_preprocess_prog(lat_rng, lon_rng):
    def preprocess(ds):
        ds = mb.subset_vars(ds, PROG_REQUIRED_VARS, PROG_OPTIONAL_VARS, "prog")
        return sel_region_with_fallback(ds, lat_rng, lon_rng)

    return preprocess


def make_preprocess_surf(lat_rng, lon_rng):
    def preprocess(ds):
        ds = mb.subset_vars(ds, SURF_REQUIRED_VARS, SURF_OPTIONAL_VARS, "surf")
        return sel_region_with_fallback(ds, lat_rng, lon_rng)

    return preprocess


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


def compute_region_budget_series(prog_patterns, surf_patterns, name, region):
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{name}_{region['key']}.nc"
    if cache_file.exists():
        print(f"Loading {name} {region['title']} from cache...")
        return xr.open_dataset(cache_file).load()

    lat_rng = region["lat"]
    lon_rng = region["lon"]
    print(f"Loading {name} data for {region['title']}...")
    state3d = load_serial_subset_patterns(
        prog_patterns,
        make_preprocess_prog(lat_rng, lon_rng),
        f"{name} prog {region['key']}",
    )
    flux2d = load_serial_subset_patterns(
        surf_patterns,
        make_preprocess_surf(lat_rng, lon_rng),
        f"{name} surf {region['key']}",
    )

    state3d, flux2d, align_freq = mb.align_time_axes(state3d, flux2d, f"{name} {region['title']}")
    ds = xr.merge([state3d, flux2d], join="inner", compat="override")
    ds = ds.sel(time=slice(ANALYSIS_START, ANALYSIS_END))
    if ds.time.size == 0:
        raise ValueError(f"No data inside analysis window for {name} {region['title']}.")

    print(
        f"  -> Synchronized {len(ds.time)} steps for {name} {region['title']} "
        f"using {align_freq} alignment"
    )

    lev_dim = mb.get_lev_dim(ds)
    has_delp = any(n in ds for n in ("DELP", "delp", "Dp", "dP"))
    p_coord = ds[lev_dim]
    lev_is_hpa = bool(p_coord.max() > 100 and p_coord.max() < 2000)

    dp = mb.build_dp_positional(ds, ds["PS"]).clip(min=0)
    if not has_delp and lev_is_hpa:
        dp = dp * 100.0

    p_mid_vals = ds[lev_dim].values
    if lev_is_hpa:
        p_mid_vals = p_mid_vals * 100.0
    p_mid_4d = xr.DataArray(p_mid_vals, dims=(lev_dim,)).broadcast_like(ds["QV"])
    below_ground = p_mid_4d > ds["PS"]

    q = ds["QV"].where(~below_ground)

    col_W = (q * dp / mb.g).sum(lev_dim, skipna=True)
    dWdt = col_W.differentiate("time", datetime_unit="s")

    Evap = mb.ensure_atm_positive(ds["LHFX"]) / mb.Lv

    precip_var = [n for n in ("PRECTOT", "TPREC", "precip") if n in ds.variables]
    if precip_var:
        Precip = ds[precip_var[0]]
    else:
        Precip = xr.zeros_like(Evap)
        print(f"  WARNING: precipitation variable not found for {name} {region['title']}.")

    StorageRelease = -dWdt
    MoistureConvergence = Precip - Evap - StorageRelease

    fields = {
        "dWdt": dWdt,
        "col_W": col_W,
        "Precip": Precip,
        "Evap": Evap,
        "MoistureConvergence": MoistureConvergence,
        "StorageRelease": StorageRelease,
    }

    series = mb.make_series_dataset(
        {name_: mb.box_mean(da) for name_, da in fields.items()},
        {
            "series_kind": "box",
            "region_name": region["title"],
            "region_key": region["key"],
            "lat_min": lat_rng[0],
            "lat_max": lat_rng[1],
            "lon_min": lon_rng[0],
            "lon_max": lon_rng[1],
            "experiment_name": name,
        },
    )
    series.to_netcdf(cache_file)
    return series


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


def trimmed_values(values, trim_fraction=ROBUST_TRIM_FRACTION):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    if n == 0:
        return values
    if trim_fraction <= 0.0 or n < 5:
        return values
    k = int(np.floor(trim_fraction * n))
    if k <= 0 or 2 * k >= n:
        return values
    return np.sort(values)[k : n - k]


def sample_sem(values):
    values = trimmed_values(values)
    n = values.size
    if n <= 1:
        return 0.0
    return float(np.nanstd(values, ddof=1) / np.sqrt(n))


def build_event_moisture_series(ds, smooth_hours=BUDGET_SMOOTH_HOURS):
    steps = mb.smooth_steps_from_hours(ds, smooth_hours)
    out = xr.Dataset(attrs=dict(ds.attrs))
    out.attrs["event_smooth_hours"] = float(smooth_hours)
    out.attrs["event_smooth_steps"] = int(steps)

    out["Precip"] = mb.rolling_event_mean(ds["Precip"], steps)
    out["Evap"] = mb.rolling_event_mean(ds["Evap"], steps)
    out["col_W"] = mb.rolling_event_mean(ds["col_W"], steps)
    out["StorageRelease"] = -out["col_W"].differentiate("time", datetime_unit="s")
    out["MoistureConvergence"] = out["Precip"] - out["Evap"] - out["StorageRelease"]
    return out


def build_event_moisture_table(series, spike_indices, window_hours=EVENT_WINDOW_HOURS):
    half_steps = mb.half_window_steps_from_hours(series, window_hours)

    rows = []
    for spike_id, idx in enumerate(spike_indices, start=1):
        i0 = max(idx - half_steps, 0)
        i1 = min(idx + half_steps, series.sizes["time"] - 1)
        window = series.isel(time=slice(i0, i1 + 1))

        precip_int = mb.integrate_flux_window(window["Precip"])
        evap_int = mb.integrate_flux_window(window["Evap"])
        conv_int = mb.integrate_flux_window(window["MoistureConvergence"])
        storage_int = -(float(series["col_W"].isel(time=i1)) - float(series["col_W"].isel(time=i0)))
        rows.append(
            {
                "spike": f"S{spike_id}",
                "time": window.time.isel(time=idx - i0).values,
                "precip_mm": precip_int,
                "evap_mm": evap_int,
                "storage_mm": storage_int,
                "conv_mm": conv_int,
            }
        )

    return rows, half_steps


def collect_region_events(region, me_series, rp_series):
    me_detect = build_event_moisture_series(me_series, smooth_hours=PRECIP_SMOOTH_HOURS)
    rp_detect = build_event_moisture_series(rp_series, smooth_hours=PRECIP_SMOOTH_HOURS)
    me_budget = build_event_moisture_series(me_series, smooth_hours=BUDGET_SMOOTH_HOURS)
    rp_budget = build_event_moisture_series(rp_series, smooth_hours=BUDGET_SMOOTH_HOURS)

    reference_precip = xr.where(
        me_detect["Precip"] >= rp_detect["Precip"],
        me_detect["Precip"],
        rp_detect["Precip"],
    )
    dt_hours = mb.infer_time_step_hours(me_detect)
    min_sep_steps = max(1, int(round(MIN_PEAK_SEPARATION_HOURS / max(dt_hours, 1.0e-6))))
    spike_idx, threshold = mb.detect_precip_spikes(
        reference_precip,
        quantile=SPIKE_QUANTILE,
        min_separation=min_sep_steps,
        max_spikes=MAX_SPIKES_PER_REGION,
    )

    me_events, half_steps = build_event_moisture_table(me_budget, spike_idx, window_hours=EVENT_WINDOW_HOURS)
    rp_events, _ = build_event_moisture_table(rp_budget, spike_idx, window_hours=EVENT_WINDOW_HOURS)

    rows = []
    for event_number, idx, me_event, rp_event in zip(range(1, len(spike_idx) + 1), spike_idx, me_events, rp_events):
        i0 = max(idx - half_steps, 0)
        i1 = min(idx + half_steps, me_detect.sizes["time"] - 1)
        event_start = me_detect.time.isel(time=i0).values
        event_end = me_detect.time.isel(time=i1).values
        record = {
            "region_key": region["key"],
            "region_name": region["title"],
            "event_id": f"{region['key']}_E{event_number:02d}",
            "peak_time": np.datetime_as_string(me_detect.time.isel(time=idx).values, unit="h"),
            "event_start": np.datetime_as_string(event_start, unit="h"),
            "event_end": np.datetime_as_string(event_end, unit="h"),
            "threshold_mm_day": float(threshold * SECONDS_PER_DAY),
            "reference_peak_mm_day": float(reference_precip.isel(time=idx) * SECONDS_PER_DAY),
            "rean_peak_mm_day": float(me_detect["Precip"].isel(time=idx) * SECONDS_PER_DAY),
            "iau_peak_mm_day": float(rp_detect["Precip"].isel(time=idx) * SECONDS_PER_DAY),
        }
        for key, _ in COMPONENTS:
            record[f"rean_{key}"] = float(me_event[key])
            record[f"iau_{key}"] = float(rp_event[key])
            record[f"diff_{key}"] = float(me_event[key] - rp_event[key])
        rows.append(record)

    print(
        f"  -> {region['title']}: detected {len(rows)} spikes above the {SPIKE_QUANTILE:.2f} quantile "
        f"(threshold {threshold * SECONDS_PER_DAY:.2f} mm day-1)"
    )
    return rows


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def event_fieldnames():
    fields = [
        "region_key",
        "region_name",
        "event_id",
        "peak_time",
        "event_start",
        "event_end",
        "threshold_mm_day",
        "reference_peak_mm_day",
        "rean_peak_mm_day",
        "iau_peak_mm_day",
    ]
    for prefix in ("rean", "iau", "diff"):
        for key, _ in COMPONENTS:
            fields.append(f"{prefix}_{key}")
    return fields


def load_event_rows(path):
    rows = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = dict(row)
            for key, value in row.items():
                if key.endswith("_mm") or key.endswith("_mm_day"):
                    parsed[key] = float(value)
            rows.append(parsed)
    return rows


def current_schema_rows(rows):
    keep = set(event_fieldnames())
    return [{key: row[key] for key in event_fieldnames() if key in row and key in keep} for row in rows]


def summarize_components(rows, region_name=None):
    summary = []
    for key, label in COMPONENTS:
        me_vals = np.array([row[f"rean_{key}"] for row in rows], dtype=float)
        rp_vals = np.array([row[f"iau_{key}"] for row in rows], dtype=float)
        diff_vals = np.array([row[f"diff_{key}"] for row in rows], dtype=float)
        summary.append(
            {
                "region_name": region_name or "ALL",
                "component_key": key,
                "component_label": label,
                "n_events": int(len(diff_vals)),
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


def plot_summary(pacific_rows, pacific_summary):
    labels = [label for _, label in COMPONENTS]

    mean_rean = np.array([row["rean_mean_mm"] for row in pacific_summary], dtype=float)
    mean_iau = np.array([row["iau_mean_mm"] for row in pacific_summary], dtype=float)

    rean_yerr = np.vstack(
        [
            np.array([row["rean_sem_mm"] for row in pacific_summary], dtype=float),
            np.array([row["rean_sem_mm"] for row in pacific_summary], dtype=float),
        ]
    )
    iau_yerr = np.vstack(
        [
            np.array([row["iau_sem_mm"] for row in pacific_summary], dtype=float),
            np.array([row["iau_sem_mm"] for row in pacific_summary], dtype=float),
        ]
    )

    x = np.arange(len(labels))
    width = 0.42
    pair_offset = 0.24
    colors = ["black", "tab:blue", "tab:green", "tab:purple", "firebrick"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 6.5))
    fig.subplots_adjust(wspace=0.25)

    ax = axes[0]
    ax.bar(x - pair_offset, mean_rean, width=width, color="navy", alpha=0.9, yerr=rean_yerr, capsize=4, label="Dynamically Imbalanced")
    ax.bar(x + pair_offset, mean_iau, width=width, color="darkorange", alpha=0.9, yerr=iau_yerr, capsize=4, label="Dynamically Balanced")
    ax.axhline(0.0, color="0.4", linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"mm per {EVENT_WINDOW_LABEL}")
    ax.set_title("(a) Mean spike-window moisture budget", loc="left", fontweight="bold", fontsize=12)
    ax.legend(loc="upper left", frameon=False)

    ax = axes[1]
    pacific_mean_diff = np.array([row["diff_mean_mm"] for row in pacific_summary], dtype=float)
    pacific_diff_yerr = np.vstack(
        [
            np.array([row["diff_sem_mm"] for row in pacific_summary], dtype=float),
            np.array([row["diff_sem_mm"] for row in pacific_summary], dtype=float),
        ]
    )
    bars = ax.bar(x, pacific_mean_diff, width=width, color=colors, alpha=0.88, yerr=pacific_diff_yerr, capsize=4)
    bars[-1].set_hatch("//")
    bars[-1].set_edgecolor("firebrick")
    bars[-1].set_linewidth(1.3)
    ax.axhline(0.0, color="0.4", linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"Difference [mm per {EVENT_WINDOW_LABEL}]")
    ax.set_title("(b) Ensemble mean difference (imbalanced - balanced)", loc="left", fontweight="bold", fontsize=12)
    label_scale = np.abs(pacific_mean_diff) + pacific_diff_yerr.max(axis=0)
    label_pad = 0.5
    if np.isfinite(label_scale).any():
        label_pad = max(label_pad, 0.04 * np.nanmax(label_scale))
    y_for_limits = [0.0]
    for xpos, row, value in zip(x, pacific_summary, pacific_mean_diff):
        if value >= 0.0:
            yloc = value + pacific_diff_yerr[1, xpos] + label_pad
            va = "bottom"
        else:
            yloc = value - pacific_diff_yerr[0, xpos] - label_pad
            va = "top"
        y_for_limits.extend(
            [
                value - pacific_diff_yerr[0, xpos],
                value + pacific_diff_yerr[1, xpos],
                yloc,
            ]
        )
        ax.text(
            xpos,
            yloc,
            f"p={row['pvalue']:.3f}",
            ha="center",
            va=va,
            fontsize=9,
        )

    finite_y = np.asarray(y_for_limits, dtype=float)
    finite_y = finite_y[np.isfinite(finite_y)]
    if finite_y.size:
        ymin = float(np.nanmin(finite_y))
        ymax = float(np.nanmax(finite_y))
        yrange = max(ymax - ymin, 1.0)
        margin = max(0.75, 0.08 * yrange)
        ax.set_ylim(ymin - margin, ymax + margin)

    for ax in axes:
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.tick_params(axis="both", labelsize=10)

    fig.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {FIGURE_PATH}")


def print_component_summary(summary):
    print("\nOverall multi-event budget summary")
    for row in summary:
        print(
            f"  {row['component_label']}: "
            f"Reanalysis={row['rean_mean_mm']:6.2f} +/- {row['rean_sem_mm']:6.2f}, "
            f"IAU={row['iau_mean_mm']:6.2f} +/- {row['iau_sem_mm']:6.2f}, "
            f"diff={row['diff_mean_mm']:6.2f} +/- {row['diff_sem_mm']:6.2f}, "
            f"p={row['pvalue']:.4f}"
        )


def main():
    event_fields = event_fieldnames()
    if EVENT_TABLE_PATH.exists():
        print(f"Loading saved spike-event statistics from {EVENT_TABLE_PATH}...")
        all_rows = current_schema_rows(load_event_rows(EVENT_TABLE_PATH))
        write_csv(EVENT_TABLE_PATH, all_rows, event_fields)
    else:
        me_prog_patterns = monthly_patterns(mb.me_prog)
        me_surf_patterns = monthly_patterns(mb.me_surf)
        rp_prog_patterns = monthly_patterns(mb.rp_prog)
        rp_surf_patterns = monthly_patterns(mb.rp_surf)

        all_rows = []
        for region in REGIONS:
            me_series = compute_region_budget_series(me_prog_patterns, me_surf_patterns, "Reanalysis-IC", region)
            rp_series = compute_region_budget_series(rp_prog_patterns, rp_surf_patterns, "IAU-IC", region)
            me_series, rp_series = xr.align(me_series, rp_series, join="inner")
            region_rows = collect_region_events(region, me_series, rp_series)
            all_rows.extend(region_rows)

        write_csv(EVENT_TABLE_PATH, all_rows, event_fields)
        print(f"Event table saved to {EVENT_TABLE_PATH}")

    region_summary_rows = []
    for region in REGIONS:
        region_rows = [row for row in all_rows if row["region_key"] == region["key"]]
        region_summary_rows.extend(summarize_components(region_rows, region_name=region["title"]))

    pacific_rows = [row for row in all_rows if row["region_key"] in PACIFIC_REGION_KEYS]
    pacific_summary = summarize_components(pacific_rows, region_name="Tropical Pacific")
    print("\nTropical Pacific ensemble-mean difference summary")
    for row in pacific_summary:
        print(
            f"  {row['component_label']}: "
            f"diff={row['diff_mean_mm']:6.2f} +/- {row['diff_sem_mm']:6.2f}, "
            f"p={row['pvalue']:.4f}"
        )

    summary_fields = [
        "region_name",
        "component_key",
        "component_label",
        "n_events",
        "rean_mean_mm",
        "rean_sem_mm",
        "iau_mean_mm",
        "iau_sem_mm",
        "diff_mean_mm",
        "diff_sem_mm",
        "pvalue",
    ]
    write_csv(SUMMARY_PATH, pacific_summary, summary_fields)
    write_csv(REGION_SUMMARY_PATH, region_summary_rows, summary_fields)
    print(f"Overall summary saved to {SUMMARY_PATH}")
    print(f"Region summary saved to {REGION_SUMMARY_PATH}")

    plot_summary(pacific_rows, pacific_summary)


if __name__ == "__main__":
    main()
