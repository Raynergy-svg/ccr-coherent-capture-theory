"""Apply the EB-screen verdicts to the existing widernet_bls_results.csv
so the triage uses the post-screen verdicts without redoing BLS.

Reads widernet_eb_screen_results.csv (produced by the back-validation)
and adds eb_screen_verdict + sub-test values to widernet_bls_results.csv.
"""
import pandas as pd

bls = pd.read_csv("widernet_bls_results.csv")
eb  = pd.read_csv("widernet_eb_screen_results.csv")
print(f"BLS rows: {len(bls)}, EB rows: {len(eb)}")

# Strip name whitespace for safer merge
bls["_name_k"] = bls["name"].astype(str).str.strip()
eb["_name_k"]  = eb["name"].astype(str).str.strip()

cols = ["_name_k","verdict","n_pass","n_inconc","n_fail",
        "sub1_sderatio","sub2_oddeven","sub3_posfrac",
        "sub4_duration","sub5_companionR","sub6_secondary"]
# sub7 (Gaia RUWE) is optional -- only present in screens run after the
# 2026-05-30 RUWE addition. Pull it in if the column exists.
if "sub7_gaia_ruwe" in eb.columns:
    cols.append("sub7_gaia_ruwe")
eb_keep = eb[cols].rename(columns={
    "verdict":"eb_screen_verdict",
    "n_pass":"eb_n_pass", "n_inconc":"eb_n_inconc", "n_fail":"eb_n_fail",
    "sub1_sderatio":"eb_sde_ratio_2P_1P",
    "sub2_oddeven":"eb_oddeven_sigma",
    "sub3_posfrac":"eb_posfrac",
    "sub4_duration":"eb_duration_ratio",
    "sub5_companionR":"eb_companion_RJup",
    "sub6_secondary":"eb_secondary_ppm",
    "sub7_gaia_ruwe":"eb_gaia_ruwe",
})

merged = bls.merge(eb_keep, on="_name_k", how="left").drop(columns=["_name_k"])
merged.to_csv("widernet_bls_results.csv", index=False)
print(f"  merged -> widernet_bls_results.csv")
print()
print("verdicts:")
print(merged.eb_screen_verdict.value_counts(dropna=False).to_string())
