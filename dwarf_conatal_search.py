"""
Co-natal pair search among the 32 actionable nearby dwarfs.

Pull GALAH dynamics VAC for the 32, plus existing GALAH chemistry, and search
all 496 pairs for matching (actions, age, chemistry). Compare to a
permutation null to assess whether any pair is genuinely co-natal vs
chance overlap of an N=32 solar-twin population.
"""
import io, time, requests
import numpy as np, pandas as pd
from scipy import stats

DC = "https://datacentral.org.au/vo/tap/sync"

o = pd.read_csv("real_dwarf_targets.csv")
ids = o["gaiadr3_source_id"].astype("int64").tolist()
print(f"32 actionable dwarfs: pulling GALAH dynamics VAC + chemistry...")

def tap(q, retries=3):
    for a in range(retries):
        try:
            r = requests.post(DC, data={"REQUEST":"doQuery","LANG":"ADQL","FORMAT":"csv","QUERY":q}, timeout=180)
            if r.status_code == 200: return pd.read_csv(io.StringIO(r.text))
        except Exception as e: pass
        time.sleep(2**a)
    return pd.DataFrame()

idlist = ",".join(str(i) for i in ids)
dyn = tap(f"""SELECT gaiadr3_source_id, ecc, R_peri, R_ap, zmax,
                     J_R, J_Z, L_Z, Energy, U_UVW, V_UVW, W_UVW
              FROM galah_dr4.vacdynamicstable
              WHERE gaiadr3_source_id IN ({idlist})""")
print(f"dynamics VAC rows: {len(dyn)}")
# chem from mainstartable
chem = tap(f"""SELECT gaiadr3_source_id, mg_fe, si_fe, fe_h, al_fe, ca_fe, ba_fe, age, teff, logg
               FROM galah_dr4.mainstartable
               WHERE gaiadr3_source_id IN ({idlist})""")
print(f"GALAH chem rows: {len(chem)}")

d = o[["gaiadr3_source_id","main_id","hab8","dec","ra"]].merge(
    dyn, on="gaiadr3_source_id", how="left").merge(
    chem, on="gaiadr3_source_id", how="left", suffixes=("","_g"))

# show coverage
print(f"\nactions coverage:")
for c in ["J_R","J_Z","L_Z","ecc"]:
    print(f"  {c}: {d[c].notna().sum()}/32")

# distance metrics: normalised pair distance in (age, chem, actions)
# z-score each axis across the 32, then Euclidean distance
def zscore(s):
    m, sd = s.mean(), s.std(ddof=0)
    return (s-m)/sd if sd>0 else s*0

z_age = zscore(d.age)
z_fe  = zscore(d.fe_h)
z_mg  = zscore(d.mg_fe)
z_si  = zscore(d.si_fe)
z_ca  = zscore(d.ca_fe)
z_al  = zscore(d.al_fe)
z_LZ  = zscore(d.L_Z)
z_JR  = zscore(d.J_R)
z_JZ  = zscore(d.J_Z)
z_ecc = zscore(d.ecc)
chem_mat = np.column_stack([z_fe, z_mg, z_si, z_ca, z_al]).astype(float)
act_mat  = np.column_stack([z_LZ, z_JR, z_JZ, z_ecc]).astype(float)
age_vec  = z_age.astype(float).values

names = d.main_id.fillna(d.gaiadr3_source_id.astype(str)).values

# compute pairwise distances
N = len(d)
pairs = []
for i in range(N):
    for j in range(i+1, N):
        # tolerate NaN by skipping pairs with any NaN
        if np.isnan(chem_mat[i]).any() or np.isnan(chem_mat[j]).any(): continue
        if np.isnan(act_mat[i]).any()  or np.isnan(act_mat[j]).any():  continue
        if np.isnan(age_vec[i])        or np.isnan(age_vec[j]):        continue
        dchem = np.linalg.norm(chem_mat[i] - chem_mat[j])
        dact  = np.linalg.norm(act_mat[i]  - act_mat[j])
        dage  = abs(age_vec[i] - age_vec[j])
        pairs.append((i, j, dchem, dact, dage, dchem+dact+dage,
                      names[i], names[j], d.iloc[i].hab8, d.iloc[j].hab8))
P = pd.DataFrame(pairs, columns=["i","j","d_chem","d_act","d_age","d_total","name_i","name_j","hab_i","hab_j"])
print(f"\npairs evaluated (after NaN drops): {len(P)}/{N*(N-1)//2}")

# Sort by total distance, show closest pairs
P = P.sort_values("d_total").reset_index(drop=True)
print(f"\nclosest 15 pairs (z-score sum of chemistry + actions + age):")
print(P.head(15)[["name_i","name_j","d_chem","d_act","d_age","d_total"]].to_string(index=False))

# Permutation null: shuffle one star's chemistry/actions/age vs the other,
# compute the distribution of "closest pair distance" under random matching
rng = np.random.default_rng(42)
N_PERM = 500
min_d_null = []
for _ in range(N_PERM):
    perm = rng.permutation(N)
    md = np.inf
    for i in range(N):
        for j in range(i+1, N):
            ii, jj = i, perm[j]
            if ii == jj: continue
            if np.isnan(chem_mat[ii]).any() or np.isnan(chem_mat[jj]).any(): continue
            if np.isnan(act_mat[ii]).any()  or np.isnan(act_mat[jj]).any():  continue
            if np.isnan(age_vec[ii])        or np.isnan(age_vec[jj]):        continue
            dt = (np.linalg.norm(chem_mat[ii] - chem_mat[jj]) +
                  np.linalg.norm(act_mat[ii]  - act_mat[jj])  +
                  abs(age_vec[ii] - age_vec[jj]))
            if dt < md: md = dt
    min_d_null.append(md)
min_d_null = np.array(min_d_null)
real_min = P.iloc[0].d_total
print(f"\nClosest real pair total z-distance: {real_min:.3f}")
print(f"Permutation null closest-pair distance: median {np.median(min_d_null):.3f}, 5th pctile {np.percentile(min_d_null,5):.3f}")
print(f"P(null gives closest pair <= real): {(min_d_null <= real_min).mean():.3f}")
if (min_d_null <= real_min).mean() < 0.05:
    print("=> SIGNIFICANT: closest real pair is closer than 95% of permutation nulls.")
    print("   Possible co-natal candidate worth follow-up.")
else:
    print("=> Closest real pair is within the permutation null range.")
    print("   No co-natal pair candidate; consistent with chance overlap of N=32 solar twins.")

P.to_csv("dwarf_conatal_pairs.csv", index=False)
print("\nDONE")
