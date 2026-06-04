"""TESS half-orbital harmonic enrichment test on the NEA TOI catalog.

Implementation of the pre-registered test in toi_harmonic_test_preregistration.py.
See that file for hypothesis, sample selection, test statistic, and decision rule
(all sealed BEFORE this script ran).
"""
import warnings; warnings.filterwarnings("ignore")
import io, json, time
import numpy as np, pandas as pd, requests
from datetime import datetime, timezone

NEA_TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
P_HALF_ORBIT_D = 6.83          # TESS spacecraft half-orbital period
P_MIN_D        = 30.0          # pre-registered period cut (lower)
P_MAX_D        = 300.0         # pre-registered period cut (upper)
FRAC_CUT       = 0.005         # 0.5% from nearest integer harmonic = "clustered"
FP_DISPS = {"FP", "FA"}
CP_DISPS = {"CP", "KP"}
N_BOOT   = 10000
RNG_SEED = 20260530
OUT_JSON = "toi_harmonic_test_results.json"

def nea_query():
    """Pull the relevant TOI columns directly via TAP."""
    q = ("SELECT toi, toipfx, tfopwg_disp, pl_orbper "
         "FROM toi "
         f"WHERE pl_orbper > {P_MIN_D} AND pl_orbper < {P_MAX_D}")
    r = requests.post(NEA_TAP, data={
        "REQUEST": "doQuery", "LANG": "ADQL",
        "FORMAT": "csv", "QUERY": q,
    }, timeout=120)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

def frac_off_nearest_harmonic(P):
    n = max(1, round(P / P_HALF_ORBIT_D))
    return abs(P - n * P_HALF_ORBIT_D) / (n * P_HALF_ORBIT_D), int(n)

def main():
    print(f"Querying NEA TOI table for {P_MIN_D} < P < {P_MAX_D} d ...")
    df = nea_query()
    print(f"  returned {len(df)} TOI rows")

    df = df.dropna(subset=["pl_orbper", "tfopwg_disp"]).copy()
    df["disp"] = df["tfopwg_disp"].astype(str).str.strip().str.upper()
    df["frac_off"], df["n_harmonic"] = zip(*df["pl_orbper"].apply(frac_off_nearest_harmonic))
    df["clustered"] = df["frac_off"] < FRAC_CUT

    fp = df[df["disp"].isin(FP_DISPS)].copy()
    cp = df[df["disp"].isin(CP_DISPS)].copy()
    print(f"  FP-group (FP+FA):  {len(fp):>5}  TOIs")
    print(f"  CP-group (CP+KP):  {len(cp):>5}  TOIs")
    if len(fp) < 30 or len(cp) < 30:
        print("  WARNING: small group(s); statistical power limited")

    f_fp = fp["clustered"].mean() if len(fp) else float("nan")
    f_cp = cp["clustered"].mean() if len(cp) else float("nan")
    delta = f_fp - f_cp
    print(f"\nfraction clustered (within {FRAC_CUT*100:.1f}% of n*{P_HALF_ORBIT_D}d):")
    print(f"  FP-group: {f_fp:.4f}  ({fp['clustered'].sum()} of {len(fp)})")
    print(f"  CP-group: {f_cp:.4f}  ({cp['clustered'].sum()} of {len(cp)})")
    print(f"  Delta   = {delta:+.4f}  (FP - CP)")

    # Bootstrap p-value: under H0, both groups are draws from a single pooled
    # population; resample labels and recompute Delta to build the null.
    rng = np.random.default_rng(RNG_SEED)
    pool_clust = np.concatenate([fp["clustered"].values, cp["clustered"].values])
    n_fp = len(fp)
    deltas_null = np.empty(N_BOOT)
    for i in range(N_BOOT):
        perm = rng.permutation(len(pool_clust))
        fp_b = pool_clust[perm[:n_fp]].mean()
        cp_b = pool_clust[perm[n_fp:]].mean()
        deltas_null[i] = fp_b - cp_b
    p_two_sided = float(np.mean(np.abs(deltas_null) >= abs(delta)))

    # Also bootstrap the Delta CI (sampling with replacement WITHIN each group)
    delta_boot = np.empty(N_BOOT)
    fp_arr = fp["clustered"].values; cp_arr = cp["clustered"].values
    for i in range(N_BOOT):
        b1 = rng.choice(fp_arr, size=len(fp_arr), replace=True).mean() if len(fp_arr) else 0
        b2 = rng.choice(cp_arr, size=len(cp_arr), replace=True).mean() if len(cp_arr) else 0
        delta_boot[i] = b1 - b2
    ci_lo, ci_med, ci_hi = np.percentile(delta_boot, [2.5, 50, 97.5])

    # Pre-registered decision rule (NO new criteria added post-hoc)
    if delta > 0 and p_two_sided < 0.01:
        decision = "CONFIRMED"
    elif abs(delta) < 0.02 and p_two_sided > 0.20:
        decision = "REFUTED"
    else:
        decision = "INCONCLUSIVE"

    print()
    print(f"null permutation test (10000 label permutations):")
    print(f"  two-sided p-value: {p_two_sided:.5f}")
    print(f"  null Delta distribution: median={np.median(deltas_null):+.4f}  "
          f"95% CI [{np.percentile(deltas_null,2.5):+.4f}, {np.percentile(deltas_null,97.5):+.4f}]")
    print(f"  bootstrap Delta CI: median={ci_med:+.4f}  95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print()
    print(f"DECISION (per pre-registered rule): {decision}")

    results = {
        "test_date_utc": datetime.now(timezone.utc).isoformat(),
        "P_min_d": P_MIN_D, "P_max_d": P_MAX_D,
        "frac_cut": FRAC_CUT, "P_half_orbit_d": P_HALF_ORBIT_D,
        "n_TOIs_total_in_window": int(len(df)),
        "n_fp_group": int(len(fp)),
        "n_cp_group": int(len(cp)),
        "f_fp_clustered": float(f_fp),
        "f_cp_clustered": float(f_cp),
        "delta": float(delta),
        "delta_bootstrap_ci": [float(ci_lo), float(ci_med), float(ci_hi)],
        "p_two_sided_perm": float(p_two_sided),
        "decision_per_prereg": decision,
        "n_boot": int(N_BOOT),
        "rng_seed": int(RNG_SEED),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {OUT_JSON}")
    # Also save the underlying contingency for transparency
    pd.DataFrame({
        "group": ["FP+FA", "CP+KP"],
        "n_total": [len(fp), len(cp)],
        "n_clustered": [int(fp["clustered"].sum()), int(cp["clustered"].sum())],
        "frac_clustered": [f_fp, f_cp],
    }).to_csv("toi_harmonic_test_contingency.csv", index=False)
    print(f"saved toi_harmonic_test_contingency.csv")

if __name__ == "__main__":
    main()
