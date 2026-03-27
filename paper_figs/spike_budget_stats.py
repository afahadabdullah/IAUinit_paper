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
PRECIP_SMOOTH_HOURS = 6.0
BUDGET_SMOOTH_HOURS = 0.0
MAX_SPIKES_PER_REGION = 999
BOOTSTRAP_SAMPLES = 20000
RNG_SEED = 42
ENSEMBLE_EQUIVALENT_FACTOR = 10
ROBUST_TRIM_FRACTION = 0.10

CACHE_DIR = Path(__file__).with_name("cache_multi_spike_v1")
EVENT_TABLE_PATH = Path(__file__).with_name("multi_spike_budget_events.csv")
SUMMARY_PATH = Path(__file__).with_name("multi_spike_budget_summary.csv")
REGION_SUMMARY_PATH = Path(__file__).with_name("multi_spike_budget_region_summary.csv")
FIGURE_PATH = Path(__file__).with_name("multi_spike_budget_stats.png")

REGIONS = [
    {"key": "wtp", "title": "Western Tropical Pacific", "lon": (143.0, 143.0), "lat": (-1.0, -1.0)},
    {"key": "ctp", "title": "Central Tropical Pacific", "lon": (-145.0, -145.0), "lat": (8.0, 8.0)},
    {"key": "etp", "title": "Eastern Tropical Pacific", "lon": (-127.0, -127.0), "lat": (7.0, 7.0)},
    {"key": "itf", "title": "Indonesian Throughflow", "lon": (114.0, 115.0), "lat": (5.0, 6.0)},
    {"key": "tio", "title": "Tropical Indian Ocean", "lon": (89.0, 92.0), "lat": (7.0, 8.0)},
    {"key": "tao", "title": "Tropical Atlantic Ocean", "lon": (-39.0, -38.0), "lat": (7.0, 8.0)},
]

COMPONENTS = [
    ("precip_jm2", r"$\int L_vP\,dt$"),
    ("evap_jm2", r"$\int E\,dt$"),
    ("conv_jm2", r"$\int MC\,dt$"),
    ("storage_jm2", r"$-\Delta\langle L_vq \rangle$"),
    ("closed_jm2", "Closed"),
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
        ds = mb.subset_vars(ds, mb.PROG_REQUIRED_VARS, mb.PROG_OPTIONAL_VARS, "prog")
        return sel_region_with_fallback(ds, lat_rng, lon_rng)

    return preprocess


def make_preprocess_surf(lat_rng, lon_rng):
    def preprocess(ds):
        ds = mb.subset_vars(ds, mb.SURF_REQUIRED_VARS, mb.SURF_OPTIONAL_VARS, "surf")
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
    p_mid_4d = xr.DataArray(p_mid_vals, dims=(lev_dim,)).broadcast_like(ds["T"])
    below_ground = p_mid_4d > ds["PS"]

    T = ds["T"].where(~below_ground)
    q = ds["QV"].where(~below_ground)
    Phi = mb.g * ds["H"].where(~below_ground)

    q_lat = mb.Lv * q
    col_L = (q_lat * dp / mb.g).sum(lev_dim, skipna=True)
    s_dry = mb.cp * T + Phi
    col_s = (s_dry * dp / mb.g).sum(lev_dim, skipna=True)
    col_h = col_s + col_L

    dMSEdt = col_h.differentiate("time", datetime_unit="s")
    dDSEdt = col_s.differentiate("time", datetime_unit="s")
    dLatentdt = col_L.differentiate("time", datetime_unit="s")

    SWTNET = ds["SWTNET"]
    OLR = ds["OLR"]
    SWGNET = ds["SWGNET"]
    LWS_dn = ds["LWS"]

    lwup_names = [n for n in ("LWUP", "LWUP_SFC", "LWGUP", "LWSUP", "LWGEM") if n in ds.variables]
    if lwup_names:
        LWUP_sfc = ds[lwup_names[0]]
    else:
        Tsurf = mb.find_surface_temp(ds, ds)
        LWUP_sfc = mb.sigma * Tsurf**4

    LW_net_sfc = LWS_dn - LWUP_sfc
    SW_net_sfc = SWGNET
    R_net_sfc = SW_net_sfc + LW_net_sfc
    R_net_toa = SWTNET - OLR
    R_col = R_net_toa - R_net_sfc

    LHF = mb.ensure_atm_positive(ds["LHFX"])
    SHF = mb.ensure_atm_positive(ds["SHFX"])

    precip_var = [n for n in ("PRECTOT", "TPREC", "precip") if n in ds.variables]
    if precip_var:
        Precip = ds[precip_var[0]] * mb.Lv
    else:
        Precip = xr.zeros_like(R_col)
        print(f"  WARNING: precipitation variable not found for {name} {region['title']}.")

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


def collect_region_events(region, me_series, rp_series):
    me_detect = mb.build_event_budget_series(me_series, smooth_hours=PRECIP_SMOOTH_HOURS)
    rp_detect = mb.build_event_budget_series(rp_series, smooth_hours=PRECIP_SMOOTH_HOURS)
    me_budget = mb.build_event_budget_series(me_series, smooth_hours=BUDGET_SMOOTH_HOURS)
    rp_budget = mb.build_event_budget_series(rp_series, smooth_hours=BUDGET_SMOOTH_HOURS)

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

    me_events, half_steps = mb.build_event_budget_table(me_budget, spike_idx, window_hours=EVENT_WINDOW_HOURS)
    rp_events, _ = mb.build_event_budget_table(rp_budget, spike_idx, window_hours=EVENT_WINDOW_HOURS)

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
            "threshold_wm2": float(threshold),
            "reference_peak_wm2": float(reference_precip.isel(time=idx)),
            "rean_peak_wm2": float(me_detect["Precip"].isel(time=idx)),
            "iau_peak_wm2": float(rp_detect["Precip"].isel(time=idx)),
        }
        for key, _ in COMPONENTS:
            record[f"rean_{key}"] = float(me_event[key])
            record[f"iau_{key}"] = float(rp_event[key])
            record[f"diff_{key}"] = float(me_event[key] - rp_event[key])
        record["rean_residual_jm2"] = float(me_event["residual_jm2"])
        record["iau_residual_jm2"] = float(rp_event["residual_jm2"])
        record["diff_residual_jm2"] = float(me_event["residual_jm2"] - rp_event["residual_jm2"])
        rows.append(record)

    print(
        f"  -> {region['title']}: detected {len(rows)} spikes above the {SPIKE_QUANTILE:.2f} quantile "
        f"(threshold {threshold:.2f} W m-2)"
    )
    return rows


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_event_rows(path):
    rows = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = dict(row)
            for key, value in row.items():
                if key.endswith("_wm2") or key.endswith("_jm2"):
                    parsed[key] = float(value)
            rows.append(parsed)
    return rows


def summarize_components(rows, region_name=None):
    summary = []
    for key, label in COMPONENTS:
        me_vals = np.array([row[f"rean_{key}"] for row in rows], dtype=float) * 1.0e-6
        rp_vals = np.array([row[f"iau_{key}"] for row in rows], dtype=float) * 1.0e-6
        diff_vals = np.array([row[f"diff_{key}"] for row in rows], dtype=float) * 1.0e-6
        summary.append(
            {
                "region_name": region_name or "ALL",
                "component_key": key,
                "component_label": label,
                "n_events": int(len(diff_vals)),
                "rean_mean_mj": float(np.nanmean(me_vals)),
                "rean_sem_mj": sample_sem(me_vals),
                "iau_mean_mj": float(np.nanmean(rp_vals)),
                "iau_sem_mj": sample_sem(rp_vals),
                "diff_mean_mj": float(np.nanmean(diff_vals)),
                "diff_sem_mj": sample_sem(diff_vals),
                "pvalue": paired_wilcoxon_pvalue(diff_vals),
            }
        )
    return summary


def summarize_across_regions(region_summary_rows):
    summary = []
    for key, label in COMPONENTS:
        comp_rows = [row for row in region_summary_rows if row["component_key"] == key]
        rean_vals = np.array([row["rean_mean_mj"] for row in comp_rows], dtype=float)
        iau_vals = np.array([row["iau_mean_mj"] for row in comp_rows], dtype=float)
        diff_vals = np.array([row["diff_mean_mj"] for row in comp_rows], dtype=float)
        summary.append(
            {
                "region_name": "REGION_MEAN",
                "component_key": key,
                "component_label": label,
                "n_events": int(len(diff_vals)),
                "rean_mean_mj": float(np.nanmean(rean_vals)),
                "rean_sem_mj": sample_sem(rean_vals),
                "iau_mean_mj": float(np.nanmean(iau_vals)),
                "iau_sem_mj": sample_sem(iau_vals),
                "diff_mean_mj": float(np.nanmean(diff_vals)),
                "diff_sem_mj": sample_sem(diff_vals),
                "pvalue": paired_wilcoxon_pvalue(diff_vals),
            }
        )
    return summary


def plot_summary(all_rows, overall_summary, region_overall_summary):
    labels = [label for _, label in COMPONENTS]
    ensemble_equivalent_total = ENSEMBLE_EQUIVALENT_FACTOR * len(all_rows)

    mean_rean = np.array([row["rean_mean_mj"] for row in overall_summary], dtype=float)
    mean_iau = np.array([row["iau_mean_mj"] for row in overall_summary], dtype=float)
    mean_diff = np.array([row["diff_mean_mj"] for row in overall_summary], dtype=float)

    rean_yerr = np.vstack(
        [
            np.array([row["rean_sem_mj"] for row in overall_summary], dtype=float),
            np.array([row["rean_sem_mj"] for row in overall_summary], dtype=float),
        ]
    )
    iau_yerr = np.vstack(
        [
            np.array([row["iau_sem_mj"] for row in overall_summary], dtype=float),
            np.array([row["iau_sem_mj"] for row in overall_summary], dtype=float),
        ]
    )
    diff_yerr = np.vstack(
        [
            np.array([row["diff_sem_mj"] for row in overall_summary], dtype=float),
            np.array([row["diff_sem_mj"] for row in overall_summary], dtype=float),
        ]
    )

    x = np.arange(len(labels))
    width = 0.36
    colors = ["black", "tab:blue", "tab:green", "tab:purple", "firebrick"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 6.5))
    fig.subplots_adjust(wspace=0.25, top=0.79, left=0.08, right=0.98, bottom=0.14)
    fig.suptitle(
        "Multi-event spike-window moisture budget statistics\n"
        "All detected spikes across the six spike_pr regions",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.82,
        (
            f"Detected regional spike windows = {len(all_rows)}; "
            f"10-member equivalent total = {ensemble_equivalent_total}"
        ),
        ha="center",
        va="center",
        fontsize=10,
    )

    ax = axes[0]
    ax.bar(x - width / 2, mean_rean, width=width, color="navy", alpha=0.9, yerr=rean_yerr, capsize=4, label="Reanalysis IC")
    ax.bar(x + width / 2, mean_iau, width=width, color="darkorange", alpha=0.9, yerr=iau_yerr, capsize=4, label="IAU IC")
    ax.axhline(0.0, color="0.4", linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MJ m$^{-2}$")
    ax.set_title("(a) Mean spike-window moisture budget", loc="left", fontweight="bold", fontsize=12)
    ax.legend(loc="upper left", frameon=False)

    ax = axes[1]
    region_mean_diff = np.array([row["diff_mean_mj"] for row in region_overall_summary], dtype=float)
    region_diff_yerr = np.vstack(
        [
            np.array([row["diff_sem_mj"] for row in region_overall_summary], dtype=float),
            np.array([row["diff_sem_mj"] for row in region_overall_summary], dtype=float),
        ]
    )
    bars = ax.bar(x, region_mean_diff, color=colors, alpha=0.88, yerr=region_diff_yerr, capsize=4)
    bars[-1].set_hatch("//")
    bars[-1].set_edgecolor("firebrick")
    bars[-1].set_linewidth(1.3)
    ax.axhline(0.0, color="0.4", linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Reanalysis - IAU\n[MJ m$^{-2}$]")
    ax.set_title("(b) Region-mean paired difference with Wilcoxon p-values", loc="left", fontweight="bold", fontsize=12)
    for xpos, row, value in zip(x, region_overall_summary, region_mean_diff):
        if value >= 0.0:
            yloc = value + region_diff_yerr[1, xpos] + 2.0
            va = "bottom"
        else:
            yloc = value - region_diff_yerr[0, xpos] - 2.0
            va = "top"
        ax.text(
            xpos,
            yloc,
            f"p={row['pvalue']:.3f}",
            ha="center",
            va=va,
            fontsize=9,
        )

    for ax in axes:
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.tick_params(axis="both", labelsize=10)

    plt.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {FIGURE_PATH}")


def print_component_summary(summary):
    print("\nOverall multi-event budget summary")
    for row in summary:
        print(
            f"  {row['component_label']}: "
            f"Reanalysis={row['rean_mean_mj']:6.2f} +/- {row['rean_sem_mj']:6.2f}, "
            f"IAU={row['iau_mean_mj']:6.2f} +/- {row['iau_sem_mj']:6.2f}, "
            f"diff={row['diff_mean_mj']:6.2f} +/- {row['diff_sem_mj']:6.2f}, "
            f"p={row['pvalue']:.4f}"
        )


def main():
    if EVENT_TABLE_PATH.exists():
        print(f"Loading saved spike-event statistics from {EVENT_TABLE_PATH}...")
        all_rows = load_event_rows(EVENT_TABLE_PATH)
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

        event_fields = [
            "region_key",
            "region_name",
            "event_id",
            "peak_time",
            "event_start",
            "event_end",
            "threshold_wm2",
            "reference_peak_wm2",
            "rean_peak_wm2",
            "iau_peak_wm2",
        ]
        for prefix in ("rean", "iau", "diff"):
            for key, _ in COMPONENTS:
                event_fields.append(f"{prefix}_{key}")
        event_fields.extend(["rean_residual_jm2", "iau_residual_jm2", "diff_residual_jm2"])
        write_csv(EVENT_TABLE_PATH, all_rows, event_fields)
        print(f"Event table saved to {EVENT_TABLE_PATH}")

    region_summary_rows = []
    for region in REGIONS:
        region_rows = [row for row in all_rows if row["region_key"] == region["key"]]
        region_summary_rows.extend(summarize_components(region_rows, region_name=region["title"]))

    overall_summary = summarize_components(all_rows)
    region_overall_summary = summarize_across_regions(region_summary_rows)
    print_component_summary(overall_summary)
    print("\nAcross-region paired-difference summary")
    for row in region_overall_summary:
        print(
            f"  {row['component_label']}: "
            f"diff={row['diff_mean_mj']:6.2f} +/- {row['diff_sem_mj']:6.2f}, "
            f"p={row['pvalue']:.4f}"
        )

    summary_fields = [
        "region_name",
        "component_key",
        "component_label",
        "n_events",
        "rean_mean_mj",
        "rean_sem_mj",
        "iau_mean_mj",
        "iau_sem_mj",
        "diff_mean_mj",
        "diff_sem_mj",
        "pvalue",
    ]
    write_csv(SUMMARY_PATH, overall_summary, summary_fields)
    write_csv(REGION_SUMMARY_PATH, region_summary_rows, summary_fields)
    print(f"Overall summary saved to {SUMMARY_PATH}")
    print(f"Region summary saved to {REGION_SUMMARY_PATH}")

    plot_summary(all_rows, overall_summary, region_overall_summary)


if __name__ == "__main__":
    main()
