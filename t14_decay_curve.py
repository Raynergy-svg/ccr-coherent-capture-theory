#!/usr/bin/env python3
"""
T14 — Chemical Coherence Decay Curve
======================================
Certan (2026) | CCT | GALAH DR4 x Cantat-Gaudin 2020

Maps C/O scatter as a continuous function of cluster age.
Fits decay models to extract chemical coherence lifetime tau.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

INPUT_FILE   = "t9_cluster_stats_with_age.csv"
PLOT_FILE    = "t14_decay_curve_plot.png"
RESULTS_FILE = "t14_results.txt"

DISK_HEATING_TAU = (1.0, 2.0)  # Gyr, Binney & Tremaine

def exp_decay(t, sigma0, A, tau):
    return sigma0 + A * (1.0 - np.exp(-t / tau))

def power_law(t, sigma0, A, alpha):
    return sigma0 + A * np.power(t + 0.01, alpha)

def main():
    out = []
    def info(msg):
        line = "[INFO] " + str(msg)
        print(line, flush=True)
        out.append(line)

    info("=" * 72)
    info("T14  Chemical Coherence Decay Curve")
    info("Certan (2026) | CCT | GALAH DR4 x CG2020")
    info("=" * 72)

    df = pd.read_csv(INPUT_FILE)
    info("Loaded " + str(len(df)) + " clusters from " + INPUT_FILE)

    # Filter
    df = df.dropna(subset=["age_gyr", "C_O_std"])
    df = df[df["N"] >= 3].copy()
    info("After filtering (valid age, N>=3): " + str(len(df)) + " clusters")

    ages = df["age_gyr"].values
    scatter = df["C_O_std"].values

    info(f"Age range: {ages.min():.3f} - {ages.max():.2f} Gyr")
    info(f"C/O std range: {scatter.min():.4f} - {scatter.max():.4f}")

    # Spearman correlation
    rho, sp_p = stats.spearmanr(ages, scatter)
    info(f"\nSpearman correlation (age vs C/O std): rho = {rho:.4f}, p = {sp_p:.2e}")
    if sp_p < 0.05 and rho > 0:
        info("  => SIGNIFICANT positive correlation: coherence degrades with age")
    elif sp_p < 0.05 and rho < 0:
        info("  => SIGNIFICANT negative correlation: older clusters MORE coherent (selection effect?)")
    else:
        info("  => Not significant at p < 0.05")

    # Fit 1: Exponential decay
    info("-" * 60)
    info("Model 1: Exponential decay  sigma(t) = sigma0 + A*(1 - exp(-t/tau))")
    try:
        p0_exp = [0.02, 0.10, 1.0]
        popt_exp, pcov_exp = curve_fit(exp_decay, ages, scatter, p0=p0_exp,
                                        bounds=([0, 0, 0.01], [0.5, 1.0, 20.0]), maxfev=10000)
        sigma0_e, A_e, tau_e = popt_exp
        perr_exp = np.sqrt(np.diag(pcov_exp))
        resid_exp = scatter - exp_decay(ages, *popt_exp)
        ss_res_exp = np.sum(resid_exp**2)
        k_exp = 3
        n = len(ages)
        aic_exp = n * np.log(ss_res_exp / n) + 2 * k_exp
        bic_exp = n * np.log(ss_res_exp / n) + k_exp * np.log(n)
        info(f"  sigma0 = {sigma0_e:.4f} +/- {perr_exp[0]:.4f}")
        info(f"  A      = {A_e:.4f} +/- {perr_exp[1]:.4f}")
        info(f"  tau    = {tau_e:.3f} +/- {perr_exp[2]:.3f} Gyr")
        info(f"  AIC = {aic_exp:.1f}, BIC = {bic_exp:.1f}")
        exp_ok = True
        if DISK_HEATING_TAU[0] <= tau_e <= DISK_HEATING_TAU[1]:
            info(f"  *** tau = {tau_e:.2f} Gyr MATCHES disk heating timescale ({DISK_HEATING_TAU[0]}-{DISK_HEATING_TAU[1]} Gyr) ***")
        elif tau_e < DISK_HEATING_TAU[0]:
            info(f"  tau = {tau_e:.2f} Gyr is SHORTER than disk heating ({DISK_HEATING_TAU[0]}-{DISK_HEATING_TAU[1]} Gyr)")
        else:
            info(f"  tau = {tau_e:.2f} Gyr is LONGER than disk heating ({DISK_HEATING_TAU[0]}-{DISK_HEATING_TAU[1]} Gyr)")
    except Exception as e:
        info(f"  Fit failed: {e}")
        exp_ok = False
        aic_exp = bic_exp = np.inf
        tau_e = np.nan

    # Fit 2: Power law
    info("-" * 60)
    info("Model 2: Power law  sigma(t) = sigma0 + A * t^alpha")
    try:
        p0_pw = [0.02, 0.05, 0.3]
        popt_pw, pcov_pw = curve_fit(power_law, ages, scatter, p0=p0_pw,
                                      bounds=([0, 0, 0.01], [0.5, 1.0, 3.0]), maxfev=10000)
        sigma0_p, A_p, alpha_p = popt_pw
        perr_pw = np.sqrt(np.diag(pcov_pw))
        resid_pw = scatter - power_law(ages, *popt_pw)
        ss_res_pw = np.sum(resid_pw**2)
        k_pw = 3
        aic_pw = n * np.log(ss_res_pw / n) + 2 * k_pw
        bic_pw = n * np.log(ss_res_pw / n) + k_pw * np.log(n)
        info(f"  sigma0 = {sigma0_p:.4f} +/- {perr_pw[0]:.4f}")
        info(f"  A      = {A_p:.4f} +/- {perr_pw[1]:.4f}")
        info(f"  alpha  = {alpha_p:.4f} +/- {perr_pw[2]:.4f}")
        info(f"  AIC = {aic_pw:.1f}, BIC = {bic_pw:.1f}")
        pw_ok = True
    except Exception as e:
        info(f"  Fit failed: {e}")
        pw_ok = False
        aic_pw = bic_pw = np.inf

    # Fit 3: Sliding Mann-Whitney step function
    info("-" * 60)
    info("Model 3: Sliding Mann-Whitney step function scan")
    age_bounds = np.arange(0.2, 3.1, 0.1)
    mw_results = []
    for ab in age_bounds:
        young = scatter[ages < ab]
        old = scatter[ages >= ab]
        if len(young) >= 5 and len(old) >= 5:
            u, p = stats.mannwhitneyu(young, old, alternative="less")
            mw_results.append({"age_bound": ab, "U": u, "p": p,
                               "n_young": len(young), "n_old": len(old),
                               "young_median": np.median(young), "old_median": np.median(old)})

    mw_df = pd.DataFrame(mw_results)
    if len(mw_df) > 0:
        best_idx = mw_df["p"].idxmin()
        best = mw_df.loc[best_idx]
        info(f"  Best step at age = {best['age_bound']:.1f} Gyr")
        info(f"    p = {best['p']:.4e}, U = {best['U']:.0f}")
        info(f"    Young median = {best['young_median']:.4f} (n={int(best['n_young'])})")
        info(f"    Old median   = {best['old_median']:.4f} (n={int(best['n_old'])})")
        if best['p'] < 0.05:
            info(f"    => SIGNIFICANT step: young clusters have lower C/O scatter")
        else:
            info(f"    => Not significant at p < 0.05")

    # Best model
    info("-" * 60)
    info("Model comparison:")
    models = []
    if exp_ok: models.append(("Exponential", aic_exp, bic_exp))
    if pw_ok:  models.append(("Power law", aic_pw, bic_pw))
    for name, aic, bic in models:
        info(f"  {name:>15}: AIC = {aic:.1f}, BIC = {bic:.1f}")
    if models:
        best_model = min(models, key=lambda x: x[1])
        info(f"  Best model (lowest AIC): {best_model[0]}")

    # Binned analysis
    info("-" * 60)
    info("Binned analysis:")
    bins = [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 10.0)]
    info(f"  {'Bin (Gyr)':>12}  {'N':>4}  {'Mean std':>9}  {'Median std':>11}  {'Std of std':>11}")
    for lo, hi in bins:
        mask = (ages >= lo) & (ages < hi)
        vals = scatter[mask]
        if len(vals) > 0:
            info(f"  {lo:.1f} - {hi:.1f}     {len(vals):>4}  {vals.mean():.4f}     {np.median(vals):.4f}       {vals.std():.4f}")
        else:
            info(f"  {lo:.1f} - {hi:.1f}        0      --          --            --")

    # Summary
    info("")
    info("=" * 72)
    info("T14 SUMMARY")
    info("=" * 72)
    info(f"Spearman rho = {rho:.4f}, p = {sp_p:.2e}")
    if exp_ok:
        info(f"Exponential tau = {tau_e:.3f} Gyr (coherence lifetime)")
        info(f"Birth coherence floor sigma0 = {sigma0_e:.4f}")
    if len(mw_df) > 0:
        info(f"Best step boundary = {best['age_bound']:.1f} Gyr (p = {best['p']:.4e})")
    info(f"Disk heating timescale reference: {DISK_HEATING_TAU[0]}-{DISK_HEATING_TAU[1]} Gyr")

    with open(RESULTS_FILE, "w") as f:
        f.write("\n".join(out))
    info(f"Saved: {RESULTS_FILE}")

    # PLOT (4 panels)
    info("Generating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("T14 Chemical Coherence Decay Curve | Certan (2026) | GALAH DR4 x CG2020",
                 fontsize=14, fontweight="bold", y=0.98)

    # Panel 1: Scatter + fitted curves
    ax1 = axes[0, 0]
    ax1.scatter(ages, scatter, s=15, alpha=0.4, color="gray", edgecolors="none", rasterized=True)
    t_fit = np.linspace(0.001, ages.max(), 500)
    if exp_ok:
        ax1.plot(t_fit, exp_decay(t_fit, *popt_exp), "r-", lw=2,
                 label=f"Exp: tau={tau_e:.2f} Gyr")
    if pw_ok:
        ax1.plot(t_fit, power_law(t_fit, *popt_pw), "b--", lw=2,
                 label=f"Power: alpha={alpha_p:.2f}")
    ax1.set_xlabel("Cluster Age (Gyr)", fontsize=11)
    ax1.set_ylabel("C/O Std Dev", fontsize=11)
    ax1.set_title(f"Age vs C/O Scatter (Spearman rho={rho:.3f}, p={sp_p:.2e})", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.15)

    # Panel 2: Residuals
    ax2 = axes[0, 1]
    if exp_ok:
        ax2.scatter(ages, resid_exp, s=15, alpha=0.4, color="red", edgecolors="none", label="Exponential", rasterized=True)
        ax2.axhline(0, color="black", lw=1)
        ax2.set_xlabel("Cluster Age (Gyr)", fontsize=11)
        ax2.set_ylabel("Residual (data - model)", fontsize=11)
        ax2.set_title("Exponential Model Residuals", fontsize=11, fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.15)

    # Panel 3: Sliding MWU p-value
    ax3 = axes[1, 0]
    if len(mw_df) > 0:
        ax3.semilogy(mw_df["age_bound"], mw_df["p"], "k-o", markersize=4, lw=1.5)
        ax3.axhline(0.05, color="red", ls="--", lw=1, label="p = 0.05")
        ax3.axhline(0.01, color="orange", ls="--", lw=1, alpha=0.7, label="p = 0.01")
        ax3.axvline(best["age_bound"], color="green", ls=":", lw=2,
                    label=f"Best step = {best['age_bound']:.1f} Gyr")
        ax3.set_xlabel("Age Boundary (Gyr)", fontsize=11)
        ax3.set_ylabel("Mann-Whitney p-value", fontsize=11)
        ax3.set_title("Sliding Step Function Scan", fontsize=11, fontweight="bold")
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.15)

    # Panel 4: Binned boxplot
    ax4 = axes[1, 1]
    bin_data = []
    bin_labels = []
    for lo, hi in bins:
        mask = (ages >= lo) & (ages < hi)
        vals = scatter[mask]
        if len(vals) > 0:
            bin_data.append(vals)
            bin_labels.append(f"{lo:.0f}-{hi:.0f}\n(n={len(vals)})")
    if bin_data:
        bp = ax4.boxplot(bin_data, labels=bin_labels, patch_artist=True,
                         medianprops=dict(color="white", linewidth=2))
        colors_box = plt.cm.viridis(np.linspace(0.2, 0.8, len(bin_data)))
        for patch, c in zip(bp["boxes"], colors_box):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        ax4.set_xlabel("Age Bin (Gyr)", fontsize=11)
        ax4.set_ylabel("C/O Std Dev", fontsize=11)
        ax4.set_title("C/O Scatter by Age Bin", fontsize=11, fontweight="bold")
        ax4.grid(True, alpha=0.15, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(PLOT_FILE, dpi=200, bbox_inches="tight")
    plt.close()
    info(f"Saved: {PLOT_FILE}")
    info("T14 complete.")

if __name__ == "__main__":
    main()
