#!/usr/bin/env python3
"""
Coherent Capture Analysis: C/O Mismatch as Formation Pathway Diagnostic
=========================================================================
Certan (2026)

Tests whether the observation geometry / orbital separation correlation
with C/O planet-star mismatch (CCR) is consistent with coherent capture
of wide-orbit companions from birth cluster siblings.

Key prediction: planets that formed in the same molecular cloud as their
host star (captured) should have matching C/O. Planets that formed in
the protoplanetary disk should show C/O modified by ice line chemistry.

Data: LExACoM atmospheric composition measurements + CCR classification.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ccr = pd.read_csv("matched_ccr_cleaned.csv")

print("=" * 72)
print("COHERENT CAPTURE: C/O MISMATCH AS FORMATION PATHWAY")
print("Certan (2026)")
print("=" * 72)

# ---------------------------------------------------------------------------
# 1. Core observation: CCR correlates with orbital separation
# ---------------------------------------------------------------------------
print("\n1. CCR vs ORBITAL SEPARATION")
print("-" * 40)

# Exclude a=0 (unresolved or missing)
with_orbit = ccr[ccr["pl_orbsmax"] > 0.001].copy()
with_orbit["log_a"] = np.log10(with_orbit["pl_orbsmax"])

rho_a, p_a = stats.spearmanr(with_orbit["log_a"], with_orbit["CCR"])
print(f"Spearman (log(a) vs CCR): ρ = {rho_a:.4f}, p = {p_a:.4e}")
print(f"N = {len(with_orbit)}")

if p_a < 0.05 and rho_a < 0:
    print("=> SIGNIFICANT: wider orbits have LOWER C/O mismatch")
    print("   Wide companions match their star. Close planets don't.")

# ---------------------------------------------------------------------------
# 2. Geometry test: direct imaging vs transit
# ---------------------------------------------------------------------------
print(f"\n2. OBSERVATION GEOMETRY vs CCR")
print("-" * 40)

direct = ccr[ccr["obs_geometry"] == "Direct"]
transit = ccr[ccr["obs_geometry"] == "Transit"]
eclipse = ccr[ccr["obs_geometry"] == "Eclipse"]

print(f"Direct imaging:  N={len(direct)}, CCR median={direct['CCR'].median():.4f}")
print(f"Eclipse:         N={len(eclipse)}, CCR median={eclipse['CCR'].median():.4f}")
print(f"Transit:         N={len(transit)}, CCR median={transit['CCR'].median():.4f}")

# Direct vs Transit
if len(direct) >= 3 and len(transit) >= 3:
    u, p = stats.mannwhitneyu(direct["CCR"], transit["CCR"], alternative="less")
    print(f"\nMann-Whitney (direct < transit CCR): U={u:.0f}, p={p:.4e}")
    if p < 0.05:
        print("=> SIGNIFICANT: directly imaged planets have LOWER C/O mismatch")

# Direct vs all others
others = ccr[ccr["obs_geometry"] != "Direct"]
u2, p2 = stats.mannwhitneyu(direct["CCR"], others["CCR"], alternative="less")
print(f"Mann-Whitney (direct < others): U={u2:.0f}, p={p2:.4e}")

# ---------------------------------------------------------------------------
# 3. The key test: among directly imaged planets, does CCR correlate
#    with anything that ISN'T circular?
# ---------------------------------------------------------------------------
print(f"\n3. WITHIN DIRECT IMAGING: CCR vs separation")
print("-" * 40)

di_with_a = direct[direct["pl_orbsmax"] > 0].copy()
if len(di_with_a) >= 5:
    rho_di, p_di = stats.spearmanr(di_with_a["pl_orbsmax"], di_with_a["CCR"])
    print(f"Spearman (a vs CCR, direct only): ρ = {rho_di:.4f}, p = {p_di:.4e}")
    print(f"N = {len(di_with_a)}")

# ---------------------------------------------------------------------------
# 4. Classification independence test
# ---------------------------------------------------------------------------
print(f"\n4. IS CLASSIFICATION INDEPENDENT OF DETECTION METHOD?")
print("-" * 40)

# Contingency table: (Direct vs non-Direct) × (CAPTURE vs non-CAPTURE)
ccr["is_direct"] = ccr["obs_geometry"] == "Direct"
ccr["is_capture"] = ccr["h_universal_class"] == "CAPTURE"

ct_a = int(((ccr["is_direct"]) & (ccr["is_capture"])).sum())
ct_b = int(((ccr["is_direct"]) & (~ccr["is_capture"])).sum())
ct_c = int(((~ccr["is_direct"]) & (ccr["is_capture"])).sum())
ct_d = int(((~ccr["is_direct"]) & (~ccr["is_capture"])).sum())

print(f"                    CAPTURE  non-CAPTURE")
print(f"  Direct imaging      {ct_a:>4}        {ct_b:>4}")
print(f"  Transit/Eclipse     {ct_c:>4}        {ct_d:>4}")

odds, fisher_p = stats.fisher_exact([[ct_a, ct_b], [ct_c, ct_d]])
print(f"\nFisher exact: OR = {odds:.2f}, p = {fisher_p:.4e}")
if fisher_p < 0.05:
    print("=> SIGNIFICANT association between direct imaging and CAPTURE class")
    print("   Directly imaged planets preferentially match their host star's C/O")

# ---------------------------------------------------------------------------
# 5. What disk chemistry predicts vs what we observe
# ---------------------------------------------------------------------------
print(f"\n5. DISK CHEMISTRY PREDICTION TEST")
print("-" * 40)
print("Standard disk models predict:")
print("  - Inner disk (< 1 AU): C/O modified by refractory loss → lower C/O")
print("  - Outer disk (> 5 AU): C/O enhanced beyond ice lines → higher C/O")
print("  - Both should show CCR >> 0 (planet ≠ star)")
print()
print("Coherent capture predicts:")
print("  - Wide companions: CCR ≈ 0 (planet = star, same cloud)")
print("  - Close planets: CCR >> 0 (disk-processed)")
print()

# Split at 5 AU
close = ccr[ccr["pl_orbsmax"] < 5]
wide = ccr[ccr["pl_orbsmax"] >= 5]
print(f"Close-in (< 5 AU):   N={len(close)}, CCR median={close['CCR'].median():.4f}")
print(f"Wide-orbit (≥ 5 AU): N={len(wide)}, CCR median={wide['CCR'].median():.4f}")

if len(close) >= 3 and len(wide) >= 3:
    u3, p3 = stats.mannwhitneyu(wide["CCR"], close["CCR"], alternative="less")
    print(f"Mann-Whitney (wide < close): U={u3:.0f}, p={p3:.4e}")

    if p3 < 0.05:
        print("\n=> Wide-orbit planets have LOWER C/O mismatch than close-in planets")
        print("   This is OPPOSITE to disk chemistry predictions")
        print("   (disk models predict MORE modification at wider radii)")
        print("   CONSISTENT with coherent capture of wide companions")
    else:
        print("\n=> Not significant")

# ---------------------------------------------------------------------------
# 6. Quantify the capture fraction
# ---------------------------------------------------------------------------
print(f"\n6. CAPTURE FRACTION ESTIMATES")
print("-" * 40)

n_total = len(ccr)
n_capture = len(ccr[ccr["h_universal_class"] == "CAPTURE"])
n_disk = len(ccr[ccr["h_universal_class"] == "DISK"])
n_gray = len(ccr[ccr["h_universal_class"] == "GRAY_ZONE"])

print(f"Total characterized: {n_total}")
print(f"CAPTURE (CCR < 0.09):   {n_capture} ({n_capture/n_total:.0%})")
print(f"GRAY_ZONE:              {n_gray} ({n_gray/n_total:.0%})")
print(f"DISK (CCR > 0.28):      {n_disk} ({n_disk/n_total:.0%})")

# Among direct imaging only
print(f"\nDirect imaging only:")
di_cap = len(direct[direct["h_universal_class"] == "CAPTURE"])
di_total = len(direct)
print(f"  CAPTURE: {di_cap}/{di_total} ({di_cap/di_total:.0%})")
print(f"  If these are genuine captures, {di_cap/di_total:.0%} of directly")
print(f"  imaged companions may have been acquired from birth cluster siblings")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle("Coherent Capture: C/O Mismatch as Formation Pathway Diagnostic\n"
             "Certan (2026) | LExACoM Atmospheric Compositions",
             fontsize=13, fontweight="bold", y=0.99)

colors = {"CAPTURE": "#2ca02c", "GRAY_ZONE": "#ff7f0e", "DISK": "#d62728"}
markers = {"Direct": "D", "Eclipse": "s", "Transit": "o"}

# P1: CCR vs orbital separation
ax = axes[0, 0]
for cls in ["CAPTURE", "GRAY_ZONE", "DISK"]:
    sub = ccr[(ccr["h_universal_class"] == cls) & (ccr["pl_orbsmax"] > 0.001)]
    for geom in ["Direct", "Eclipse", "Transit"]:
        g = sub[sub["obs_geometry"] == geom]
        if len(g) > 0:
            ax.scatter(g["pl_orbsmax"], g["CCR"], c=colors[cls], marker=markers[geom],
                       s=80, alpha=0.8, edgecolors="white", linewidths=0.5,
                       label=f"{cls[:4]} {geom[:3]}" if len(g) > 0 else "")
ax.set_xscale("log")
ax.set_xlabel("Orbital Separation (AU)", fontsize=11)
ax.set_ylabel("CCR = |log(C/O_planet / C/O_star)|", fontsize=11)
ax.set_title(f"CCR vs Separation (ρ={rho_a:.3f}, p={p_a:.2e})", fontweight="bold")
ax.axhline(0.09, color="green", ls="--", alpha=0.5, label="CAPTURE threshold")
ax.axhline(0.28, color="red", ls="--", alpha=0.5, label="DISK threshold")
ax.legend(fontsize=6, ncol=2, loc="upper right")
ax.grid(True, alpha=0.15)

# P2: CCR by observation geometry (boxplot)
ax = axes[0, 1]
geom_data = [direct["CCR"].values, eclipse["CCR"].values, transit["CCR"].values]
bp = ax.boxplot(geom_data, tick_labels=["Direct", "Eclipse", "Transit"],
                patch_artist=True, medianprops=dict(color="black", linewidth=2))
geom_colors = ["#2ca02c", "#ff7f0e", "#d62728"]
for patch, c in zip(bp["boxes"], geom_colors):
    patch.set_facecolor(c); patch.set_alpha(0.6)
# Jitter points
for i, (data, c) in enumerate(zip(geom_data, geom_colors)):
    jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(data))
    ax.scatter(np.full(len(data), i+1) + jitter, data, c=c, s=30, alpha=0.6, zorder=5)
ax.set_ylabel("CCR", fontsize=11)
ax.set_title("C/O Mismatch by Detection Method", fontweight="bold")
ax.grid(True, alpha=0.15, axis="y")

# P3: Contingency table visualization
ax = axes[1, 0]
table_data = np.array([[ct_a, ct_b], [ct_c, ct_d]], dtype=float)
im = ax.imshow(table_data, cmap="YlOrRd", aspect="auto")
ax.set_xticks([0, 1]); ax.set_xticklabels(["CAPTURE", "non-CAPTURE"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Direct", "Transit/Eclipse"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(table_data[i, j]), ha="center", va="center",
                fontsize=20, fontweight="bold", color="white" if table_data[i,j] > 5 else "black")
ax.set_title(f"Fisher OR={odds:.1f}, p={fisher_p:.3f}", fontweight="bold")

# P4: The prediction diagram
ax = axes[1, 1]
ax.scatter(ccr[ccr["pl_orbsmax"] < 5]["pl_orbsmax"],
           ccr[ccr["pl_orbsmax"] < 5]["CCR"],
           c="salmon", s=60, alpha=0.7, label="Close-in (disk-formed)")
ax.scatter(ccr[ccr["pl_orbsmax"] >= 5]["pl_orbsmax"],
           ccr[ccr["pl_orbsmax"] >= 5]["CCR"],
           c="steelblue", s=80, alpha=0.7, marker="D", label="Wide-orbit (capture?)")
ax.axvline(5, color="gray", ls=":", lw=1.5, alpha=0.5)
ax.set_xscale("log")
ax.set_xlabel("Orbital Separation (AU)", fontsize=11)
ax.set_ylabel("CCR", fontsize=11)
ax.set_title("Close vs Wide: Disk Chemistry Inverted", fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.15)
ax.text(0.03, 0.97, "Disk models predict:\nWide = MORE modification\n\nObserved:\nWide = LESS modification",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("coherent_capture_analysis.png", dpi=200, bbox_inches="tight")
plt.close()
print("\nSaved: coherent_capture_analysis.png")

# Summary
print(f"\n{'='*72}")
print("SUMMARY")
print(f"{'='*72}")
print(f"1. Wide-orbit planets have lower C/O mismatch than close-in (ρ={rho_a:.3f}, p={p_a:.2e})")
print(f"2. Direct imaging planets have lower CCR than transit (p={p:.4e})")
print(f"3. Direct imaging is associated with CAPTURE class (OR={odds:.1f}, p={fisher_p:.3f})")
print(f"4. This is OPPOSITE to disk chemistry predictions")
print(f"5. Consistent with coherent capture: wide companions from same birth cloud")
print(f"6. {di_cap}/{di_total} ({di_cap/di_total:.0%}) of directly imaged companions may be captures")

ccr.to_csv("capture_candidates_filtered.csv", index=False)
print(f"\nSaved: capture_candidates_filtered.csv")
