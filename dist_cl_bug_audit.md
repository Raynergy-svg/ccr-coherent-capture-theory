# Audit: `dist_cl` units bug in the T9 cluster pipeline

**Severity: high — it changes two headline paper results (T10 and T19).**
Reproduction in [`audit_dist_cl.py`](audit_dist_cl.py); root-cause fix in [`t9_cluster_coherence.py`](t9_cluster_coherence.py).

## The bug

`t9_cluster_coherence.py` selected the cluster distance column by substring match:

```python
dist_col = next((c for c in cg20.columns
                 if "dist" in c.lower() or "plx" in c.lower()), None)
```

The Cantat-Gaudin 2020 catalog (`J/A+A/640/A1`) exposes **both** `plx` (parallax, mas) **and** `DistPc` (distance, pc). `next()` returned `plx`, so **`dist_cl` is the cluster parallax in mas** — then every downstream script treats it as a distance in **kpc** (`SkyCoord(..., distance=dist_cl * u.kpc)`; `t10` additionally does `dist *= 1000` "kpc→pc").

Because distance ≈ 1/parallax, the bug **inverts the near/far ordering**: nearby clusters are placed far away and vice-versa.

## Proof

`dist_cl` equals the published CG2020 parallax for every spot-checked cluster:

| Cluster | stored `dist_cl` | CG2020 plx (mas) | true distance |
|---|---|---|---|
| Melotte 22 (Pleiades) | 7.346 | 7.36 | 136 pc |
| NGC 2632 (Praesepe) | 5.361 | 5.36 | 187 pc |
| Blanco 1 | 4.210 | 4.20 | 238 pc |
| Ruprecht 147 | 3.250 | 3.25 | 308 pc |
| Teutsch 80 | 0.349 | 0.349 | 2865 pc |

Stored range is 0.036–21.06 — absurd as kpc, exactly right as mas. The Pleiades (closest real cluster) got the *largest* "distance."

## Impact (faithfully re-running T19 and T10 on corrected distances)

Corrected distance: `d_kpc = 1 / dist_cl` (= 1/parallax). The buggy column reproduces the published numbers exactly, which validates the replication.

| Test | Metric | Published (buggy) | **Corrected** | Effect |
|---|---|---|---|---|
| **T19** | Spearman(R_gal, C/O scatter) | ρ = −0.228, p = 3.4×10⁻⁹ | **ρ = −0.091, p = 0.020** | direction survives, **~2.5× weaker**, significance 10⁻⁹→10⁻² |
| **T19** | Mann-Whitney inner vs outer | p = 2.1×10⁻⁸ | p = 7.9×10⁻³ | outer still more coherent, much weaker |
| **T10** | Mantel r (spatial vs chemical) | r = −0.010, p = 0.60 | **r = +0.067, p = 0.014** | **conclusion flips** |

- **T19** ("outer disk more coherent"): the gradient is **real but much weaker** than published. ρ ≈ −0.09 (p ≈ 0.02), not −0.23 (p ≈ 10⁻⁹). The qualitative claim holds; the quoted effect size and significance do not.
- **T10** ("chemical coherence is spatially independent, p = 0.60"): **does not survive.** With correct distances the Mantel test is **positive and significant** (r = +0.067, p = 0.014) — chemically similar clusters are weakly *spatially closer*. This is the opposite of the published conclusion and needs to be corrected in the paper.

Other scripts consuming `dist_cl` as kpc are also affected and should be re-run: **T16b** (dissolved-member 3D positions), **T20c** (NGC 6253), and any plot/table using cluster distances. T10's `dist.mean() < 50 → ×1000` heuristic is *correct once* `dist_cl` is true kpc, so no change is needed there after the source fix.

## The fix

`t9_cluster_coherence.py` now selects the distance-in-pc column explicitly and stores `dist_cl` in kpc, falling back to `1/parallax` only if no distance column exists:

```python
dist_pc_col = next((c for c in cg20.columns
                    if c.lower() in ("distpc","dist","dist50","distance")
                    or ("dist" in c.lower() and "pc" in c.lower())), None)
plx_col = next((c for c in cg20.columns if c.lower() in ("plx","parallax")), None)
...
if dist_pc_col is not None:
    cg20["dist_cl"] = pd.to_numeric(cg20[dist_pc_col], errors="coerce") / 1000.0   # pc -> kpc
elif plx_col is not None:
    cg20["dist_cl"] = 1.0 / pd.to_numeric(cg20[plx_col], errors="coerce")          # mas -> kpc
```

## What still needs doing (cannot run here — needs the 723 MB GALAH FITS)

1. Regenerate `t9_matched_stars.csv` (and `t9_cluster_stats_*`) by re-running the fixed `t9_cluster_coherence.py` against `galah_dr4_allstar_240705.fits`.
2. Re-run **t10, t16b, t19, t20c** on the regenerated CSVs (they read CSVs only, so this is quick once the CSVs are correct).
3. Update the paper: T10's "spatially independent (p=0.60)" claim reverses to a weak positive correlation (p≈0.01); T19's gradient weakens to ρ≈−0.09 (p≈0.02). The README summary table rows for T10 and T19 need revision.

> Note: I left the canonical result files (`t10_mantel_results.txt`, `t19_results.txt`) and result CSVs untouched so the current published state is preserved. The audit script writes nothing. I can patch the existing CSVs with `1/plx` distances if you want corrected downstream outputs *before* a full FITS re-run.

## Aside — co-natal cluster search for HD 28888 (the other half of the request)

Searched Hunt & Reffert 2023 and Cantat-Gaudin 2020 for a surviving cluster HD 28888 (6.3 Gyr, [Fe/H] +0.13, near-circular thin-disk orbit, R_peri 6.3 / R_apo 8.8 kpc, |z|<65 pc) could trace to:

- 165 clusters older than 4 Gyr exist; only ~20 lie within 1 kpc and near the plane.
- Closest age-matches: HSC_2028 (292 pc, 6.79 Gyr) and HSC_1234 (388 pc, 6.07 Gyr) — but their proper motions/RVs don't match HD 28888, and **neither catalog carries [Fe/H]**, so chemical confirmation is impossible from them.
- M67 (NGC 2682, 4.3 Gyr) sits at Z ≈ +470 pc, off-plane — excluded.

**No surviving open cluster is a viable birth site.** This is expected: a 6.3 Gyr cluster has almost certainly dissolved, and over that timescale kinematic coherence is largely erased by radial migration and disk heating. Tracing HD 28888 to its birth siblings would require the chemical-tagging precision the paper itself reserves for the 4MOST era.
