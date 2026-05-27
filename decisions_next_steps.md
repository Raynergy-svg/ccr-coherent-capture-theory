# Decisions — dist_cl fix follow-through

Current state: bug fixed in code; CSVs patched (kpc) with backups; T19/T10 re-run into `*_distfix` files; `revisions_T10_T19.md` holds drafted edits with authoritative numbers. Nothing in `README.md` or `certan2026_cct.tex` has been changed yet — these decisions gate that.

---

## Decision 1 — How to resolve the T10 reversal

The corrected Mantel test is **r = +0.067, p = 0.013** (weak positive spatial–chemical correlation), reversing the published "spatially independent, p = 0.60." Open question: **is this a real spatial confound, or just the radial [Fe/H] gradient leaking in?** With correct distances the [Fe/H]–R_gal correlation tightened to ρ = −0.65, so metal-rich clusters now sit at similar (inner) radii — which could manufacture an apparent spatial–chemical link without any true ISM-locality effect.

| Option | What it means | Cost | Effect on paper |
|---|---|---|---|
| **1A — Investigate first (recommended)** | Run a **partial Mantel** controlling for [Fe/H] (and/or redo on C/O *residuals* after removing the radial [Fe/H] trend). If the spatial correlation vanishes, the "necessary condition for chemical tagging" is still satisfied — it was metallicity, not locality. | ~10 min, no new data | Could *restore* the original conclusion with a cleaner justification, or confirm a genuine weak confound |
| **1B — Report the weak confound as-is** | Accept r = +0.067 at face value; reframe text to "minor confound, r² < 0.5%." | none (already drafted) | Honest but leaves a small dent in the tagging argument |

**Recommendation: 1A.** It's cheap and it decides the framing — you may not need to weaken the claim at all if the correlation is purely the metallicity gradient.

---

## Decision 2 — Who applies the edits to README / paper

`revisions_T10_T19.md` has exact before/after text for all 8 `.tex` passages + 2 README rows.

| Option | What I do |
|---|---|
| **2A — Apply everything** | Edit `README.md` + `certan2026_cct.tex` directly, swap the figure to `t19_galactic_radius_distfix_plot.png`, commit. |
| **2B — Apply README only** | Update the 2 README rows; leave the DOI'd manuscript for you to edit from the draft. |
| **2C — Apply nothing (recommended until Decision 1 settles)** | Keep both as drafts; the T10 wording depends on the Decision 1 outcome. |

**Recommendation: 2C now → 2A once Decision 1 is resolved**, since the T10 sentences change depending on whether the spatial correlation survives [Fe/H] control.

---

## Decision 3 — Regenerate the FITS-dependent tests (T16b, T20c)

Both also consumed the buggy `dist_cl` but need `galah_dr4_allstar_240705.fits` (723 MB, not in the container) to re-run.

| Option | What it means |
|---|---|
| **3A — You run them locally** | Pull the branch, drop the FITS in place, run the (already-fixed) `t9_cluster_coherence.py` then `t16b`/`t20c`. |
| **3B — Provide FITS access here** | If the file is fetchable from a URL the sandbox can reach, I regenerate them this session. |
| **3C — Defer** | Leave T16b/T20c on buggy distances for now; flag in the paper as pending. |

**Recommendation: 3A** unless the FITS is easily reachable — T16b's dissolved-member recovery uses 3D positions, so its numbers may shift and should be redone before the paper is finalized.

---

## Fastest path if you don't want to micro-decide
**1A + 2A + 3A:** I run the partial-Mantel check now, apply all README/paper edits with whichever T10 framing the check supports, and you regenerate T16b/T20c locally with the FITS. Say the word and I'll start with the partial-Mantel investigation.
