"""Triage the BLS pipeline output to identify candidates worth deep follow-up.

Inputs:
  widernet_bls_results.csv -- per-target results from widernet_bls_pipeline.py

Triage logic (applied in order):
  1. SDE filter: keep only SDE >= 7
  2. Systematic-period rejection: drop periods in [0.4-0.95], [13.5-14.0],
     [27-28] (TESS spacecraft systematic windows). Already flagged by pipeline.
  3. Centroid verdict: keep only ON_TARGET (or accept MARGINAL if SDE >= 10)
  4. Quality scoring: rank survivors by an "interesting" metric:
     - more sectors  -> better baseline -> more reliable
     - lower P_d/baseline ratio -> more observed transits
     - higher SDE
     - planet-like depth (50-2000 ppm; very large depths suggest EB)
     - habitable T_eq is a bonus but not a requirement

Output:
  widernet_candidates.csv -- ranked survivors
  widernet_triage_summary.md -- human-readable summary with per-candidate
                                 quicklook stats
"""
import os, json
import numpy as np, pandas as pd

RESULTS_CSV = "widernet_bls_results.csv"
OUT_CSV = "widernet_candidates.csv"
OUT_MD = "widernet_triage_summary.md"
CAND_DIR = "widernet_candidates"

def triage(df):
    """Apply triage filters and produce ranked candidate list."""
    if "sde" not in df.columns:
        return df.head(0).assign(triage_score=0.0)

    df = df.copy()
    # parse boolean systematic flag if it's string
    if df["is_systematic_period"].dtype == object:
        df["is_systematic_period"] = df["is_systematic_period"].astype(str).str.lower().isin(["true","1"])

    # SDE filter
    s = df[df.sde.fillna(0) >= 7.0].copy()
    print(f"  SDE >= 7:                 {len(s)}")

    # Systematic period rejection
    s = s[~s.is_systematic_period.fillna(False)].copy()
    print(f"  after systematic rejection: {len(s)}")

    # Centroid verdict
    if "centroid_verdict" not in s.columns:
        s["centroid_verdict"] = ""
    on_target = s.centroid_verdict.fillna("").str.startswith("ON_TARGET")
    marginal_high_sde = s.centroid_verdict.fillna("").str.contains("MARGINAL") & (s.sde >= 10)
    s = s[on_target | marginal_high_sde].copy()
    print(f"  after centroid filter:     {len(s)}")

    # Depth filter: 50 - 5000 ppm (planet-like; <50 is noise, >5000 is likely EB)
    s = s[s.depth_ppm.between(50, 5000)].copy()
    print(f"  after depth filter (50-5000 ppm): {len(s)}")

    # Per-target sanity: require >=3 expected transits across baseline
    s["n_transits_expected"] = s.baseline_d / s.P_d
    s = s[s.n_transits_expected >= 3].copy()
    print(f"  after n_transits >= 3:     {len(s)}")

    # Triage score: combination of SDE, sectors, transits, planet plausibility
    # All factors normalized roughly 0-1 then summed
    s["score_sde"]      = np.clip((s.sde - 7) / 5, 0, 1)        # 7 -> 0, 12 -> 1
    s["score_sectors"]  = np.clip(s.n_sectors / 15, 0, 1)       # 15 sec -> 1
    s["score_n_trans"]  = np.clip(np.log10(s.n_transits_expected) / 1.5, 0, 1)
    s["score_depth"]    = np.where(s.depth_ppm < 500, 1.0,
                                    np.clip(1 - (s.depth_ppm - 500) / 3000, 0.3, 1.0))
    s["score_HZ"]       = np.where(s.T_eq_K.between(200, 400), 0.5, 0.0)  # bonus
    s["triage_score"] = (s.score_sde + s.score_sectors + s.score_n_trans +
                         s.score_depth + s.score_HZ)
    s = s.sort_values("triage_score", ascending=False).reset_index(drop=True)
    return s

def write_summary(df_cand, df_all, out_md):
    with open(out_md, "w") as f:
        f.write("# Wider-Net BLS Triage Summary\n\n")
        f.write(f"Total targets processed: {len(df_all)}\n\n")
        n_sde = (df_all.sde.fillna(0) >= 7).sum() if "sde" in df_all.columns else 0
        f.write(f"SDE >= 7 hits: {n_sde}\n\n")
        f.write(f"Candidates surviving full triage: **{len(df_cand)}**\n\n")
        f.write("---\n\n")
        if len(df_cand) == 0:
            f.write("No survivors. Either nothing real in the search space\n")
            f.write("or the triage filters are too strict.\n")
            return
        f.write("## Ranked candidates\n\n")
        f.write("| rank | name | TIC | T | n_sec | P_d | dep_ppm | SDE | R_p | T_eq | cent verdict | score |\n")
        f.write("|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|\n")
        for i, r in df_cand.iterrows():
            f.write(f"| {i+1} | {r['name']} | {r.get('tic_id','')} | "
                    f"{r.get('Tmag', float('nan')):.2f} | {int(r['n_sectors'])} | "
                    f"{r['P_d']:.4f} | {r['depth_ppm']:.0f} | {r['sde']:.2f} | "
                    f"{r.get('R_p_Rearth', float('nan')):.2f} | "
                    f"{r.get('T_eq_K', float('nan')):.0f} | "
                    f"{r.get('centroid_verdict','-')[:25]} | {r['triage_score']:.2f} |\n")
        f.write("\n\n## Per-candidate diagnostics\n\n")
        for i, r in df_cand.head(20).iterrows():
            f.write(f"### #{i+1}: {r['name']}\n\n")
            f.write(f"- TIC: {r.get('tic_id','')}\n")
            f.write(f"- Position: ({r['ra']:.5f}, {r['dec']:.5f})\n")
            f.write(f"- Tmag: {r.get('Tmag', float('nan')):.2f}\n")
            f.write(f"- R* = {r.get('R_star', float('nan')):.3f} R_sun, "
                    f"M* = {r.get('M_star', float('nan')):.3f}, "
                    f"Teff = {r.get('Teff', float('nan')):.0f} K\n")
            f.write(f"- Sectors: {int(r['n_sectors'])}, baseline = {r['baseline_d']:.0f} d\n")
            f.write(f"- BLS: P = {r['P_d']:.5f} d, depth = {r['depth_ppm']:.0f} ppm, "
                    f"dur = {r['dur_h']:.2f} h, SDE = {r['sde']:.2f}\n")
            f.write(f"- Planet (if real): R = {r.get('R_p_Rearth', float('nan')):.2f} R_Earth, "
                    f"a = {r.get('a_AU', float('nan')):.4f} AU, "
                    f"T_eq = {r.get('T_eq_K', float('nan')):.0f} K\n")
            f.write(f"- Centroid verdict: {r.get('centroid_verdict','-')}\n")
            f.write(f"- Centroid offset: {r.get('centroid_offset_arcsec', float('nan')):.3f} arcsec\n")
            f.write(f"- Centroid significance: {r.get('centroid_sigma', float('nan')):.2f} sigma\n")
            f.write(f"- Triage score: {r['triage_score']:.2f}\n")
            cand_dir = os.path.join(CAND_DIR, str(r['name']).replace("/", "_"))
            bls_png = os.path.join(cand_dir, "bls.png")
            if os.path.exists(bls_png):
                f.write(f"- Plot: `{bls_png}`\n")
            f.write("\n")

def main():
    if not os.path.exists(RESULTS_CSV):
        print(f"  {RESULTS_CSV} not found -- run widernet_bls_pipeline.py first")
        return
    df = pd.read_csv(RESULTS_CSV)
    print(f"Loaded {len(df)} target results")
    cand = triage(df)
    cand.to_csv(OUT_CSV, index=False)
    write_summary(cand, df, OUT_MD)
    print(f"\nSaved:")
    print(f"  {OUT_CSV}  ({len(cand)} candidates)")
    print(f"  {OUT_MD}")

if __name__ == "__main__":
    main()
