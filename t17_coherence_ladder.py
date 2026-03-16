#!/usr/bin/env python3
"""
T17 — The Coherence Lifetime Ladder
=====================================
Certan (2026) | Coherent Capture Theory | APOGEE DR17 OCCAM

Extends T14 (single-element C/O decay curve) using the full multi-element
coherence score from T15. Tracks how the fraction of coherent clusters
changes with age at different coherence levels:

  Level 1: C/O only
  Level 2: C/O + 1 alpha (any of Mg/Fe, Si/Fe, Al/Fe)
  Level 3: C/O + 2 alpha
  Level 4: C/O + all 3 alpha (Mg + Si + Al)
  Level 5: All 5 (C/O + Mg + Si + Al + Fe/H)

Fits survival curves to extract multi-element coherence half-lives.
Tests whether τ_multi >> τ_single — i.e., whether richer fingerprints
are more durable than single-element signals.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MATRIX_FILE   = "t15_coherence_matrix.csv"
STAR_FILE     = "tapogee_matched_stars.csv"
CLUSTER_FILE  = "tapogee_cluster_stats_with_age.csv"

RESULTS_FILE  = "t17_results.txt"
PLOT_FILE     = "t17_coherence_ladder_plot.png"
CSV_FILE      = "t17_ladder_data.csv"

# Coherence thresholds (from T15)
CO_THRESH   = 0.05
MGFE_THRESH = 0.03
SIFE_THRESH = 0.03
ALFE_THRESH = 0.05
FEH_THRESH  = 0.03

MIN_PER_BIN = 5    # minimum clusters per age bin for a data point

# Age bins (Gyr) — designed to give reasonable counts
AGE_BINS = [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 10.0)]

# Disk heating reference
DISK_HEATING_TAU = (1.0, 2.0)  # Gyr, Binney & Tremaine

out_lines = []
def info(msg):
    line = "[INFO] " + str(msg)
    print(line, flush=True)
    out_lines.append(line)

# ---------------------------------------------------------------------------
# Survival models
# ---------------------------------------------------------------------------
def exp_survival(t, f0, tau):
    """Exponential decay: f(t) = f0 * exp(-t/tau)"""
    return f0 * np.exp(-t / tau)

def stretched_exp(t, f0, tau, beta):
    """Stretched exponential: f(t) = f0 * exp(-(t/tau)^beta)"""
    return f0 * np.exp(-(t / tau)**beta)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
info("=" * 72)
info("T17  The Coherence Lifetime Ladder")
info("Certan (2026) | CCT | APOGEE DR17 OCCAM")
info("=" * 72)

matrix = pd.read_csv(MATRIX_FILE)
info(f"Loaded {len(matrix)} clusters from T15 coherence matrix")

# Ensure boolean columns
for col in ["CO_coherent", "MgFe_coh", "SiFe_coh", "AlFe_coh", "FeH_coh"]:
    if col in matrix.columns:
        matrix[col] = matrix[col].astype(str).str.lower().isin(["true", "1"])

# Filter to clusters with valid ages
aged = matrix[matrix["age_gyr"].notna()].copy()
info(f"Clusters with ages: {len(aged)}")
info(f"Age range: {aged['age_gyr'].min():.3f} - {aged['age_gyr'].max():.2f} Gyr")

# Also filter to clusters with all 5 scatter values
complete = aged.dropna(subset=["CO_std", "MgFe_std", "SiFe_std", "AlFe_std", "FeH_std"])
info(f"Complete (all 5 elements + age): {len(complete)}")

# ---------------------------------------------------------------------------
# 2. Define coherence levels
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info("Defining coherence levels...")

# Count alpha coherences
complete["n_alpha_coh"] = (
    complete["MgFe_coh"].astype(int) +
    complete["SiFe_coh"].astype(int) +
    complete["AlFe_coh"].astype(int)
)

# Level definitions
complete["L1_CO"] = complete["CO_coherent"]
complete["L2_CO_plus1"] = complete["CO_coherent"] & (complete["n_alpha_coh"] >= 1)
complete["L3_CO_plus2"] = complete["CO_coherent"] & (complete["n_alpha_coh"] >= 2)
complete["L4_CO_plus3"] = complete["CO_coherent"] & (complete["n_alpha_coh"] >= 3)
complete["L5_all5"] = (complete["CO_coherent"] & complete["MgFe_coh"] &
                       complete["SiFe_coh"] & complete["AlFe_coh"] & complete["FeH_coh"])

levels = [
    ("L1_CO",        "C/O only",          "#1f77b4"),
    ("L2_CO_plus1",  "C/O + 1α",          "#ff7f0e"),
    ("L3_CO_plus2",  "C/O + 2α",          "#2ca02c"),
    ("L4_CO_plus3",  "C/O + 3α (all)",    "#d62728"),
    ("L5_all5",      "All 5 elements",     "#9467bd"),
]

# Overall fractions
info(f"\nOverall coherence fractions ({len(complete)} clusters):")
for col, label, _ in levels:
    n = complete[col].sum()
    info(f"  {label:<20}: {n:>3}/{len(complete)} = {n/len(complete):.1%}")

# ---------------------------------------------------------------------------
# 3. Binned survival fractions
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info("Binned coherence fractions by age...")

bin_results = []

info(f"\n{'Age bin':>12}  {'N':>4}  " + "  ".join(f"{l[1]:>12}" for l in levels))
for lo, hi in AGE_BINS:
    mask = (complete["age_gyr"] >= lo) & (complete["age_gyr"] < hi)
    bin_df = complete[mask]
    n = len(bin_df)

    row = {"age_lo": lo, "age_hi": hi, "age_mid": (lo + hi) / 2.0, "N": n}

    fracs = []
    for col, label, _ in levels:
        if n >= MIN_PER_BIN:
            f = bin_df[col].sum() / n
        else:
            f = np.nan
        row[col] = f
        fracs.append(f)

    bin_results.append(row)

    frac_str = "  ".join(
        f"{f:>12.1%}" if not np.isnan(f) else f"{'--':>12}" for f in fracs
    )
    info(f"  {lo:.1f}-{hi:.1f} Gyr  {n:>4}  {frac_str}")

bin_df_all = pd.DataFrame(bin_results)

# ---------------------------------------------------------------------------
# 4. Continuous analysis: coherence score vs age
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info("Continuous analysis: multi-element coherence score vs age")

score_valid = complete[complete["coherence_score"].notna()].copy()
info(f"Clusters with coherence score + age: {len(score_valid)}")

rho_score, p_score = stats.spearmanr(score_valid["age_gyr"], score_valid["coherence_score"])
info(f"Spearman (age vs coherence_score): rho = {rho_score:.4f}, p = {p_score:.4e}")
if p_score < 0.05 and rho_score > 0:
    info("  => SIGNIFICANT: coherence score INCREASES with age (less coherent)")
elif p_score < 0.05 and rho_score < 0:
    info("  => SIGNIFICANT: coherence score DECREASES with age (more coherent — selection effect?)")
else:
    info("  => Not significant at p < 0.05")

# Also test individual element scatters vs age
info("\nElement-by-element Spearman correlations (scatter vs age):")
for col, label in [("CO_std", "C/O"), ("MgFe_std", "Mg/Fe"), ("SiFe_std", "Si/Fe"),
                    ("AlFe_std", "Al/Fe"), ("FeH_std", "Fe/H")]:
    valid = complete[[col, "age_gyr"]].dropna()
    if len(valid) >= 10:
        rho_e, p_e = stats.spearmanr(valid["age_gyr"], valid[col])
        sig = "***" if p_e < 0.001 else "**" if p_e < 0.01 else "*" if p_e < 0.05 else "n.s."
        info(f"  {label:>5} scatter vs age: rho={rho_e:+.4f}  p={p_e:.4e}  {sig}")

# ---------------------------------------------------------------------------
# 5. Fit survival curves
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info("Fitting exponential survival curves...")
info("Model: f(t) = f0 * exp(-t/τ)")

fit_results = {}

for col, label, color in levels:
    # Use bin midpoints and fractions
    valid_bins = bin_df_all[bin_df_all[col].notna()].copy()
    if len(valid_bins) < 3:
        info(f"  {label}: insufficient bins for fitting")
        continue

    t_data = valid_bins["age_mid"].values
    f_data = valid_bins[col].values

    # Weights proportional to sqrt(N)
    weights = np.sqrt(valid_bins["N"].values)

    # Skip if no decline visible
    if f_data[-1] >= f_data[0] and len(f_data) > 2:
        info(f"  {label}: no decline detected (flat or increasing)")
        fit_results[col] = {"tau": np.inf, "f0": f_data[0], "label": label,
                            "half_life": np.inf, "fit_ok": False}
        continue

    try:
        p0 = [f_data[0], 2.0]
        bounds = ([0.01, 0.05], [1.0, 50.0])
        popt, pcov = curve_fit(exp_survival, t_data, f_data, p0=p0,
                               sigma=1.0/weights, bounds=bounds, maxfev=10000)
        f0_fit, tau_fit = popt
        perr = np.sqrt(np.diag(pcov))
        half_life = tau_fit * np.log(2)

        # Goodness of fit
        f_pred = exp_survival(t_data, *popt)
        ss_res = np.sum((f_data - f_pred)**2)
        ss_tot = np.sum((f_data - f_data.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        info(f"  {label}:")
        info(f"    f0  = {f0_fit:.3f} ± {perr[0]:.3f}")
        info(f"    τ   = {tau_fit:.3f} ± {perr[1]:.3f} Gyr")
        info(f"    t½  = {half_life:.3f} Gyr")
        info(f"    R²  = {r2:.3f}")

        if DISK_HEATING_TAU[0] <= tau_fit <= DISK_HEATING_TAU[1]:
            info(f"    *** τ matches disk heating timescale ***")

        fit_results[col] = {
            "tau": tau_fit, "tau_err": perr[1], "f0": f0_fit, "f0_err": perr[0],
            "half_life": half_life, "r2": r2, "label": label, "fit_ok": True,
            "popt": popt
        }

    except Exception as e:
        info(f"  {label}: fit failed — {e}")
        fit_results[col] = {"tau": np.nan, "label": label, "half_life": np.nan,
                            "fit_ok": False}

# ---------------------------------------------------------------------------
# 6. Fit continuous coherence score decay
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info("Fitting coherence score decay curve...")
info("Model: score(t) = s0 + A*(1 - exp(-t/τ))  [score increases = less coherent]")

def score_decay(t, s0, A, tau):
    return s0 + A * (1.0 - np.exp(-t / tau))

try:
    ages_s = score_valid["age_gyr"].values
    scores = score_valid["coherence_score"].values

    p0_s = [0.15, 0.10, 2.0]
    bounds_s = ([0, 0, 0.05], [1.0, 1.0, 50.0])
    popt_s, pcov_s = curve_fit(score_decay, ages_s, scores, p0=p0_s,
                                bounds=bounds_s, maxfev=10000)
    s0_fit, A_fit, tau_score = popt_s
    perr_s = np.sqrt(np.diag(pcov_s))

    resid_s = scores - score_decay(ages_s, *popt_s)
    ss_res_s = np.sum(resid_s**2)
    ss_tot_s = np.sum((scores - scores.mean())**2)
    r2_s = 1 - ss_res_s / ss_tot_s if ss_tot_s > 0 else 0

    info(f"  s0  = {s0_fit:.4f} ± {perr_s[0]:.4f}")
    info(f"  A   = {A_fit:.4f} ± {perr_s[1]:.4f}")
    info(f"  τ   = {tau_score:.3f} ± {perr_s[2]:.3f} Gyr")
    info(f"  R²  = {r2_s:.4f}")

    score_fit_ok = True
except Exception as e:
    info(f"  Score decay fit failed: {e}")
    score_fit_ok = False
    tau_score = np.nan

# ---------------------------------------------------------------------------
# 7. Compare timescales
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info("COHERENCE LIFETIME COMPARISON")

info(f"\n{'Level':<20} {'τ (Gyr)':>10} {'t½ (Gyr)':>10} {'f0':>8}")
for col, label, _ in levels:
    if col in fit_results and fit_results[col].get("fit_ok"):
        fr = fit_results[col]
        info(f"{label:<20} {fr['tau']:>10.3f} {fr['half_life']:>10.3f} {fr['f0']:>8.3f}")
    elif col in fit_results:
        fr = fit_results[col]
        tau_str = "∞" if fr.get("tau") == np.inf else "N/A"
        info(f"{label:<20} {tau_str:>10} {'--':>10} {fr.get('f0', 0):>8.3f}")

if score_fit_ok:
    info(f"\nMulti-element score τ = {tau_score:.3f} Gyr")

info(f"\nT14 reference (C/O only, GALAH 655 clusters): τ = 1.288 Gyr")
info(f"Disk heating timescale (Binney & Tremaine): τ = 1.0–2.0 Gyr")

# Check if multi-element τ > single-element τ
fitted_taus = {col: fit_results[col]["tau"]
               for col in fit_results
               if fit_results[col].get("fit_ok") and np.isfinite(fit_results[col]["tau"])}

if len(fitted_taus) >= 2:
    tau_vals = list(fitted_taus.values())
    tau_labels = [fit_results[col]["label"] for col in fitted_taus]
    longest_idx = np.argmax(tau_vals)
    shortest_idx = np.argmin(tau_vals)
    info(f"\nLongest τ:  {tau_labels[longest_idx]} ({tau_vals[longest_idx]:.3f} Gyr)")
    info(f"Shortest τ: {tau_labels[shortest_idx]} ({tau_vals[shortest_idx]:.3f} Gyr)")
    if tau_vals[longest_idx] > 1.5 * tau_vals[shortest_idx]:
        info("=> Multi-element fingerprints have LONGER coherence lifetime")
        info("   Richer chemical fingerprints are more durable — information content")
        info("   is preserved longer in higher-dimensional abundance space.")

# ---------------------------------------------------------------------------
# 8. Bootstrap uncertainty on binned fractions
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info("Bootstrap confidence intervals on binned fractions (1000 resamples)...")

rng = np.random.default_rng(42)
N_BOOT = 1000
boot_ci = {}

for col, label, _ in levels:
    boot_ci[col] = {}
    for lo, hi in AGE_BINS:
        mask = (complete["age_gyr"] >= lo) & (complete["age_gyr"] < hi)
        bin_vals = complete.loc[mask, col].values
        n = len(bin_vals)
        if n < MIN_PER_BIN:
            boot_ci[col][(lo, hi)] = (np.nan, np.nan)
            continue

        boot_fracs = np.zeros(N_BOOT)
        for b in range(N_BOOT):
            resample = rng.choice(bin_vals, size=n, replace=True)
            boot_fracs[b] = resample.mean()
        boot_ci[col][(lo, hi)] = (np.percentile(boot_fracs, 2.5),
                                   np.percentile(boot_fracs, 97.5))

# ---------------------------------------------------------------------------
# 9. Save data
# ---------------------------------------------------------------------------
info("\n" + "-" * 60)
info("Saving outputs...")

complete.to_csv(CSV_FILE, index=False)
info(f"Saved: {CSV_FILE}")

with open(RESULTS_FILE, "w") as f:
    f.write("\n".join(out_lines))
info(f"Saved: {RESULTS_FILE}")

# ---------------------------------------------------------------------------
# 10. Generate 4-panel plot
# ---------------------------------------------------------------------------
info("Generating plot...")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle("T17 Coherence Lifetime Ladder | Certan (2026) | APOGEE DR17 OCCAM\n"
             "Multi-element coherence survival as a function of cluster age",
             fontsize=13, fontweight="bold", y=0.99)

# Panel 1: Survival curves — fraction coherent vs age
ax1 = axes[0, 0]

for col, label, color in levels:
    valid_bins = bin_df_all[bin_df_all[col].notna()].copy()
    if len(valid_bins) == 0:
        continue

    t_mid = valid_bins["age_mid"].values
    fracs = valid_bins[col].values

    # Error bars from bootstrap
    yerr_lo = []
    yerr_hi = []
    for _, row in valid_bins.iterrows():
        ci = boot_ci[col].get((row["age_lo"], row["age_hi"]), (np.nan, np.nan))
        f = row[col]
        yerr_lo.append(max(0, f - ci[0]) if not np.isnan(ci[0]) else 0)
        yerr_hi.append(max(0, ci[1] - f) if not np.isnan(ci[1]) else 0)

    ax1.errorbar(t_mid, fracs, yerr=[yerr_lo, yerr_hi],
                 fmt="o-", color=color, label=label, capsize=4,
                 markersize=8, linewidth=2, elinewidth=1.5)

    # Overlay fit if available
    if col in fit_results and fit_results[col].get("fit_ok"):
        t_fit = np.linspace(0.01, 8, 200)
        f_fit = exp_survival(t_fit, *fit_results[col]["popt"])
        ax1.plot(t_fit, f_fit, "--", color=color, alpha=0.5, linewidth=1.5)

# Disk heating band
ax1.axvspan(DISK_HEATING_TAU[0], DISK_HEATING_TAU[1], alpha=0.08, color="orange",
            label="Disk heating τ (1–2 Gyr)")

ax1.set_xlabel("Cluster Age (Gyr)", fontsize=11)
ax1.set_ylabel("Fraction Coherent", fontsize=11)
ax1.set_title("Coherence Survival Curves by Level", fontsize=11, fontweight="bold")
ax1.legend(fontsize=7.5, loc="upper right")
ax1.set_xlim(0, 8)
ax1.set_ylim(-0.05, 1.05)
ax1.grid(True, alpha=0.15)

# Panel 2: Coherence score vs age (continuous)
ax2 = axes[0, 1]
sc = ax2.scatter(score_valid["age_gyr"], score_valid["coherence_score"],
                 c=score_valid["n_alpha_coh"], cmap="RdYlGn_r",
                 s=60, alpha=0.7, edgecolors="white", linewidths=0.5,
                 vmin=0, vmax=3)
plt.colorbar(sc, ax=ax2, label="N alpha elements coherent", pad=0.02)

if score_fit_ok:
    t_fit2 = np.linspace(0.01, score_valid["age_gyr"].max(), 200)
    ax2.plot(t_fit2, score_decay(t_fit2, *popt_s), "r-", lw=2.5,
             label=f"Decay fit: τ={tau_score:.2f} Gyr", alpha=0.8)

ax2.set_xlabel("Cluster Age (Gyr)", fontsize=11)
ax2.set_ylabel("Multi-Element Coherence Score\n(lower = more coherent)", fontsize=10)
ax2.set_title(f"Coherence Score vs Age (ρ={rho_score:.3f}, p={p_score:.2e})",
              fontsize=11, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.15)

# Panel 3: Stacked bar chart — coherence level composition by age bin
ax3 = axes[1, 0]
x3 = np.arange(len(AGE_BINS))
bin_labels = [f"{lo:.1f}–{hi:.1f}" for lo, hi in AGE_BINS]

# For stacked bars: compute incremental fractions
# L5 ⊂ L4 ⊂ L3 ⊂ L2 ⊂ L1, so plot as stacked
level_cols = [l[0] for l in levels]
level_labels = [l[1] for l in levels]
level_colors = [l[2] for l in levels]

# Plot from bottom (most stringent) to top (least stringent)
# Actually, plot NOT-coherent + each level
bar_width = 0.6
prev = np.zeros(len(AGE_BINS))

for col, label, color in reversed(levels):
    vals = []
    for lo, hi in AGE_BINS:
        row = bin_df_all[(bin_df_all["age_lo"] == lo) & (bin_df_all["age_hi"] == hi)]
        if len(row) > 0 and not np.isnan(row.iloc[0][col]):
            vals.append(row.iloc[0][col])
        else:
            vals.append(0)
    vals = np.array(vals)
    ax3.bar(x3, vals, bar_width, color=color, alpha=0.8, label=label,
            edgecolor="white", linewidth=0.5)

ax3.set_xticks(x3)
ax3.set_xticklabels(bin_labels, fontsize=10)
ax3.set_xlabel("Age Bin (Gyr)", fontsize=11)
ax3.set_ylabel("Fraction Coherent", fontsize=11)
ax3.set_title("Coherence Level by Age Bin", fontsize=11, fontweight="bold")
ax3.legend(fontsize=7.5, loc="upper right", ncol=2)
ax3.set_ylim(0, 1.0)
ax3.grid(True, alpha=0.15, axis="y")

# Add N labels
for i, (lo, hi) in enumerate(AGE_BINS):
    row = bin_df_all[(bin_df_all["age_lo"] == lo) & (bin_df_all["age_hi"] == hi)]
    if len(row) > 0:
        n = row.iloc[0]["N"]
        ax3.text(i, -0.04, f"N={n:.0f}", ha="center", fontsize=8, color="gray")

# Panel 4: Half-life ladder diagram
ax4 = axes[1, 1]

fitted_levels = [(col, label, color) for col, label, color in levels
                 if col in fit_results and fit_results[col].get("fit_ok")]

if fitted_levels:
    y_pos = np.arange(len(fitted_levels))
    taus = [fit_results[col]["tau"] for col, _, _ in fitted_levels]
    tau_errs = [fit_results[col].get("tau_err", 0) for col, _, _ in fitted_levels]
    half_lives = [fit_results[col]["half_life"] for col, _, _ in fitted_levels]
    colors_bar = [color for _, _, color in fitted_levels]
    labels_bar = [label for _, label, _ in fitted_levels]

    ax4.barh(y_pos, taus, xerr=tau_errs, color=colors_bar, alpha=0.8,
             edgecolor="white", linewidth=1, height=0.5, capsize=4)

    # Disk heating band
    ax4.axvspan(DISK_HEATING_TAU[0], DISK_HEATING_TAU[1], alpha=0.1, color="orange",
                label="Disk heating τ")
    ax4.axvline(1.288, color="gray", ls=":", lw=1.5,
                label="T14 τ = 1.29 Gyr (C/O, GALAH)")

    for i, (tau, hl) in enumerate(zip(taus, half_lives)):
        ax4.text(tau + 0.15, i, f"τ={tau:.2f}  t½={hl:.2f} Gyr",
                 va="center", fontsize=9, fontweight="bold")

    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(labels_bar, fontsize=10)
    ax4.set_xlabel("Coherence Lifetime τ (Gyr)", fontsize=11)
    ax4.set_title("Coherence Half-Life Ladder", fontsize=11, fontweight="bold")
    ax4.legend(fontsize=8, loc="lower right")
    ax4.grid(True, alpha=0.15, axis="x")
    ax4.set_xlim(0, max(taus) * 1.6 if taus else 5)
else:
    ax4.text(0.5, 0.5, "Insufficient data for\nhalf-life fits",
             ha="center", va="center", transform=ax4.transAxes,
             fontsize=14, color="gray")
    ax4.set_title("Coherence Half-Life Ladder", fontsize=11, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(PLOT_FILE, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
info(f"Saved: {PLOT_FILE}")

# ---------------------------------------------------------------------------
# 11. Final Summary
# ---------------------------------------------------------------------------
info("\n" + "=" * 72)
info("T17 SUMMARY")
info("=" * 72)
info(f"Clusters analyzed: {len(complete)} (with all 5 elements + age)")
info(f"Age range: {complete['age_gyr'].min():.3f} – {complete['age_gyr'].max():.2f} Gyr")
info(f"")

info("Coherence fractions (overall):")
for col, label, _ in levels:
    n = complete[col].sum()
    info(f"  {label:<20}: {n}/{len(complete)} = {n/len(complete):.1%}")

info(f"\nCoherence score vs age: ρ = {rho_score:.4f}, p = {p_score:.4e}")
if score_fit_ok:
    info(f"Score decay τ = {tau_score:.3f} Gyr")

info(f"\nSurvival fit τ values:")
for col, label, _ in levels:
    if col in fit_results and fit_results[col].get("fit_ok"):
        fr = fit_results[col]
        info(f"  {label:<20}: τ = {fr['tau']:.3f} Gyr  (t½ = {fr['half_life']:.3f} Gyr)")
    elif col in fit_results and fit_results[col].get("tau") == np.inf:
        info(f"  {label:<20}: τ = ∞ (no decline detected)")

info(f"\nReference:")
info(f"  T14 (C/O only, GALAH):  τ = 1.288 Gyr")
info(f"  Disk heating:           τ = 1.0–2.0 Gyr")

# Key finding
if fitted_taus:
    max_tau = max(fitted_taus.values())
    min_tau = min(fitted_taus.values())
    if max_tau > 1.5 * min_tau and max_tau > 1.5:
        info(f"\n=> KEY RESULT: Multi-element coherence lifetime ({max_tau:.2f} Gyr) exceeds")
        info(f"   single-element lifetime ({min_tau:.2f} Gyr) by {max_tau/min_tau:.1f}×")
        info(f"   Higher-dimensional chemical fingerprints are MORE durable.")
        info(f"   The information content of stellar chemistry degrades more slowly")
        info(f"   when measured across multiple nucleosynthetic channels simultaneously.")
    elif max_tau > 1.0:
        info(f"\n=> RESULT: Coherence lifetimes range from {min_tau:.2f} to {max_tau:.2f} Gyr")
        info(f"   consistent with disk heating timescale.")
    else:
        info(f"\n=> RESULT: Short coherence lifetimes detected ({min_tau:.2f}–{max_tau:.2f} Gyr)")

info(f"\nOutput files:")
info(f"  {RESULTS_FILE}")
info(f"  {CSV_FILE}")
info(f"  {PLOT_FILE}")
info(f"\nT17 complete.")

# Save final results
with open(RESULTS_FILE, "w") as f:
    f.write("\n".join(out_lines))
