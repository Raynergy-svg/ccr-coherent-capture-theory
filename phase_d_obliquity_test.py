"""Phase D obliquity test of the Laws of Coherency (Nov 5, 2025).

Implementation of the test sealed in phase_d_obliquity_test_preregistration.py.
See that file for hypothesis, sample selection, statistic, and decision rule
(all locked BEFORE this script ran). DO NOT change those in this file.
"""
import warnings; warnings.filterwarnings("ignore")
import io, json, time
import numpy as np, pandas as pd, requests
from datetime import datetime, timezone

NEA_TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
OUT_JSON = "phase_d_obliquity_results.json"
OUT_CSV  = "phase_d_obliquity_systems.csv"

# Pre-registered thresholds (do not edit)
LAMBDA_THRESHOLD_DEG    = 15.0          # |lambda| > 15 = "misaligned"
SIGMA_REQUIREMENT       = 2.0           # require |lambda| > 15 at 2 sigma
HJ_PERIOD_DAYS          = 10.0          # hot-Jupiter cut (period)
HJ_MASS_MJUP            = 0.3           # hot-Jupiter cut (mass)
MIN_PLANETS_PER_HOST    = 2             # multi-planet only
F_SUPPORTED_LO          = 0.03          # decision rule lower
F_SUPPORTED_HI          = 0.15          # decision rule upper
MIN_N_FOR_DECISION      = 10            # minimum sample size for a decision

def query_nea_obliquity_table():
    """Pull every planet with a finite projected obliquity, plus what we
    need for the cuts."""
    q = ("SELECT pl_name, hostname, pl_orbper, pl_bmassj, "
         "pl_projobliq, pl_projobliqerr1, pl_projobliqerr2, pl_orbeccen, "
         "sy_pnum "
         "FROM ps "
         "WHERE pl_projobliq IS NOT NULL "
         "AND default_flag = 1")
    print(f"Querying NEA ps table for pl_projobliq...")
    r = requests.post(NEA_TAP, data={
        "REQUEST": "doQuery", "LANG": "ADQL",
        "FORMAT": "csv", "QUERY": q,
    }, timeout=180)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    print(f"  returned {len(df)} planets with measured projected obliquity")
    return df

def apply_cuts(df):
    """Apply the pre-registered sample cuts."""
    n0 = len(df)

    # cut 1: multi-planet host
    df = df[df["sy_pnum"] >= MIN_PLANETS_PER_HOST].copy()
    n_after_mp = len(df)
    print(f"  after multi-planet cut (sy_pnum >= {MIN_PLANETS_PER_HOST}): {n_after_mp} (lost {n0-n_after_mp})")

    # cut 2: hot Jupiter exclusion -- P<10d AND M>0.3 MJ
    is_hj = (df["pl_orbper"] < HJ_PERIOD_DAYS) & (df["pl_bmassj"] > HJ_MASS_MJUP)
    df = df[~is_hj].copy()
    n_after_hj = len(df)
    print(f"  after hot-Jupiter exclusion (P<{HJ_PERIOD_DAYS}d AND M>{HJ_MASS_MJUP}MJ): {n_after_hj} (lost {n_after_mp-n_after_hj} HJs)")

    # cut 3: keep only finite |lambda| with a usable uncertainty
    df["lambda_abs"] = df["pl_projobliq"].abs()
    # uncertainty: use the larger of (err1, |err2|) as conservative sigma
    sig_pos = df["pl_projobliqerr1"].abs()
    sig_neg = df["pl_projobliqerr2"].abs()
    df["lambda_sigma"] = np.fmax(sig_pos.fillna(np.inf), sig_neg.fillna(np.inf))
    # If neither error is finite, set sigma=inf so the system fails the
    # 2-sigma significance test below (cannot establish misalignment).
    n_with_err = (np.isfinite(df["lambda_sigma"])).sum()
    print(f"  with usable lambda error bars: {n_with_err}/{n_after_hj}")

    return df

def evaluate(df):
    """Apply the pre-registered decision rule."""
    # significance-aware misalignment: lambda - 2*sigma > 15 (lower edge of
    # the 2-sigma error bar above the threshold)
    df["misaligned_2sig"] = (df["lambda_abs"] - SIGMA_REQUIREMENT * df["lambda_sigma"]) > LAMBDA_THRESHOLD_DEG
    # Also record nominal-only (no significance test) for context
    df["misaligned_nominal"] = df["lambda_abs"] > LAMBDA_THRESHOLD_DEG

    n_total = len(df)
    n_misaligned_2sig = int(df["misaligned_2sig"].sum())
    n_misaligned_nominal = int(df["misaligned_nominal"].sum())
    f_2sig = n_misaligned_2sig / n_total if n_total else float("nan")
    f_nominal = n_misaligned_nominal / n_total if n_total else float("nan")

    # Decision rule (PRE-REGISTERED, do not adjust)
    if n_total < MIN_N_FOR_DECISION:
        decision = "INSUFFICIENT"
    elif n_misaligned_2sig == 0:
        decision = "REFUTED"
    elif F_SUPPORTED_LO <= f_2sig <= F_SUPPORTED_HI:
        decision = "SUPPORTED"
    elif f_2sig > F_SUPPORTED_HI:
        decision = "EXCESS"
    else:  # 0 < f_2sig < 0.03
        decision = "BELOW_SUPPORTED_RANGE"   # not pre-named, intermediate of REFUTED/SUPPORTED

    return dict(
        n_total=n_total,
        n_misaligned_2sig=n_misaligned_2sig,
        n_misaligned_nominal=n_misaligned_nominal,
        f_misaligned_2sig=f_2sig,
        f_misaligned_nominal=f_nominal,
        decision=decision,
    )

def main():
    df = query_nea_obliquity_table()
    df = apply_cuts(df)
    if len(df) == 0:
        print("\nno systems passed cuts; aborting")
        return

    summary = evaluate(df)
    print()
    print(f"Multi-planet non-hot-Jupiter systems with measured pl_projobliq: {summary['n_total']}")
    print(f"  |lambda|>15 at 2 sigma: {summary['n_misaligned_2sig']}  (fraction {summary['f_misaligned_2sig']:.3f})")
    print(f"  |lambda|>15 nominal:    {summary['n_misaligned_nominal']}  (fraction {summary['f_misaligned_nominal']:.3f})")
    print(f"  -> DECISION (pre-registered rule): {summary['decision']}")
    print()

    # Show the misaligned candidates with context (the Laws of Coherency
    # also predict e~0.05-0.10 and f_MMR<0.3 for the same systems; we don't
    # have f_MMR here per-planet but we record what we have)
    mis = df[df["misaligned_2sig"]].copy()
    if len(mis):
        print(f"Systems with |lambda|>15 at 2 sigma (candidate captures by the Laws):")
        for _, r in mis.iterrows():
            ecc = r["pl_orbeccen"]
            ecc_str = f"e={ecc:.3f}" if np.isfinite(ecc) else "e=NA"
            print(f"  {r['hostname']:<20}  {r['pl_name']:<22}  "
                  f"|lambda|={r['lambda_abs']:.1f}+/-{r['lambda_sigma']:.1f}  "
                  f"P={r['pl_orbper']:.2f}d  M={r['pl_bmassj']:.3f}MJ  {ecc_str}")
    else:
        print("(no misaligned candidates found)")

    df.to_csv(OUT_CSV, index=False)
    results = {
        "test_date_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_file": "phase_d_obliquity_test_preregistration.py",
        "lambda_threshold_deg": LAMBDA_THRESHOLD_DEG,
        "sigma_requirement": SIGMA_REQUIREMENT,
        "hot_jupiter_cut_P_days": HJ_PERIOD_DAYS,
        "hot_jupiter_cut_M_Mjup": HJ_MASS_MJUP,
        "min_planets_per_host": MIN_PLANETS_PER_HOST,
        "f_supported_range": [F_SUPPORTED_LO, F_SUPPORTED_HI],
        "min_n_for_decision": MIN_N_FOR_DECISION,
        **summary,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nsaved {OUT_JSON}")
    print(f"saved {OUT_CSV}")

if __name__ == "__main__":
    main()
