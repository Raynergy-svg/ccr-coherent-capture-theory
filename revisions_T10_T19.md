# Proposed revisions to README and paper for the corrected `dist_cl` distances

Headline corrected numbers (validated in `audit_dist_cl.py`; confirmed by the full
re-run in `t19_results_distfix.txt` / `t10_mantel_results_distfix.txt`):

| | Published (buggy) | Corrected |
|---|---|---|
| **T10** Mantel r, p | r = −0.010, p = 0.60 → "spatially independent" | **r = +0.067, p = 0.013** → weak positive correlation |
| **T10** partial r (age-controlled) | −0.009 | **+0.063** (also flips positive) |
| **T19** Spearman(R_gal, C/O scatter) | ρ = −0.228, p = 3.4×10⁻⁹ | **ρ = −0.091, p = 0.02** |
| **T19** Mann–Whitney inner vs outer | p = 2.1×10⁻⁸ (med 0.1305 vs 0.0937) | **p = 7.9×10⁻³** (med 0.1061 vs 0.0975) |
| **T19** partial ρ \| [Fe/H] | ρ = −0.288, p = 5.5×10⁻¹⁴ | **ρ = −0.147, p = 1.7×10⁻⁴** (survives) |
| **T19** Kruskal–Wallis | H = 37.0, p = 4.7×10⁻⁸ | **H = 28.8, p = 2.4×10⁻⁶** |
| **T19** Fisher OR (coherent frac) | OR = 0.16, p = 1.3×10⁻² | **OR = 0.99, p = 1.0 → NULL** |
| **T19** \|Z\| vs scatter | ρ = 0.120, p = 2.1×10⁻³ | **ρ = 0.064, p = 0.10 → NULL** |

**Interpretation guidance**
- **T19**: the gradient is *real but ~2.5× weaker*. The core tests survive — Spearman (ρ=−0.091, p=0.02), Mann–Whitney (p=7.9×10⁻³), Kruskal–Wallis (p=2.4×10⁻⁶), and the partial correlation controlling for [Fe/H] (ρ_partial=−0.147, p=1.7×10⁻⁴; note this is *stronger* than the raw Spearman because the radial [Fe/H] gradient tightened to ρ=−0.65). But two corroborating tests **go null**: the inner/outer coherent-fraction Fisher test (OR 0.16→0.99) and the |Z| test (p 0.002→0.10). So keep the gradient claim, drop the Fisher/|Z| supporting statistics, and report the weaker effect size.
- **T10**: this is the consequential one (confirmed by the full 9999-permutation re-run, `t10_mantel_results_distfix.txt`). The published null (p = 0.60) was used to argue that "chemically similar clusters are not spatial neighbours — a necessary condition for chemical tagging." With correct distances the Mantel test is **positive and formally significant** (r = +0.067, p = 0.013), and the age-controlled partial r likewise flips positive (−0.009 → +0.063). The *effect is tiny* (r² ≈ 0.004, < 0.5 % of variance; significance is driven by the 214 185 cluster pairs), but the clean "spatially independent" statement no longer holds. Recommend reframing to: *a weak residual spatial–chemical correlation exists (r = 0.067), explaining < 0.5 % of pairwise variance, so spatial position is a minor confound rather than the dominant driver of chemical similarity* — which still supports chemical tagging but honestly, not as a clean null.

---

## README.md edits

**Summary table — T10 row** (currently):
> `| **T10** | Chemical coherence is spatially independent | Mantel p = 0.60 |`

→ replace with:
> `| **T10** | Weak residual spatial–chemical correlation (r²<0.5%) | Mantel r = +0.067, p = 0.014 |`

**Summary table — T19 row** (currently):
> `| **T19** | Outer disk more coherent; persists after [Fe/H] control | Spearman p = 10⁻⁹ |`

→ replace with:
> `| **T19** | Outer disk more coherent (weak); persists after [Fe/H] control | Spearman ρ = −0.09, p = 0.02 |`

---

## paper/certan2026_cct.tex edits

### 1. Abstract, line 61
**Old:**
> `Coherence is spatially independent (Mantel $r = -0.010$, $p = 0.60$), satisfying a necessary condition for chemical tagging.`

**New:**
> `The residual spatial--chemical correlation is weak (Mantel $r = 0.067$, $p = 0.014$; $<0.5$ per cent of pairwise variance), so spatial position is at most a minor confound for chemical tagging.`

### 2. Abstract, line 65
**Old:**
> `Coherence decreases toward the inner disc ($\rho = -0.228$, $p = 3.4 \times 10^{-9}$).`

**New:**
> `Coherence decreases weakly toward the inner disc ($\rho = -0.091$, $p = 0.02$).`

### 3. Section "Chemical order is spatially independent", line 275
**Old:**
> `The Mantel test on the 655 GALAH clusters yields $r = -0.010$ with $p = 0.596$, indicating no correlation between inter-cluster chemical distance and spatial distance.`

**New:**
> `The Mantel test on the 655 GALAH clusters yields $r = 0.067$ with $p = 0.014$. Although formally significant given the $214\,185$ cluster pairs, the correlation is very weak ($r^2 < 0.5$ per cent), indicating that inter-cluster chemical distance is only marginally related to spatial distance and that spatial position is a minor confound rather than the dominant driver of chemical similarity.`
>
> *(Also revise the surrounding sentences/section heading at lines 269–273 that assert clean spatial independence; the "necessary condition" is weakly violated, not satisfied.)*

### 4. Section "Galactic radius coherence gradient", lines 420–424
**Old (line 420):**
> `The Spearman correlation between $R_{\mathrm{gal}}$ and \co scatter is $\rho = -0.228$ ($p = 3.4 \times 10^{-9}$): scatter decreases with increasing Galactocentric radius.`

**New:**
> `The Spearman correlation between $R_{\mathrm{gal}}$ and \co scatter is $\rho = -0.091$ ($p = 0.02$): scatter decreases weakly with increasing Galactocentric radius.`

**Old (line 423):** `The Kruskal--Wallis test across four radial bins yields $H = 37.0$, $p = 4.7 \times 10^{-8}$.`
**New:** ⟨FILL H, p from t19_results_distfix.txt⟩

**Old (line 424):** `Partial correlation controlling for \feh yields $\rho_{\mathrm{partial}} = -0.288$, $p = 5.5 \times 10^{-14}$, demonstrating that the gradient persists after removing the effect of the radial metallicity gradient.`
**New:** ⟨FILL ρ_partial, p; adjust "demonstrating ... persists" wording to match the corrected significance⟩

### 5. Figure caption, line 432
Replace both `$\rho = -0.228$, $p = 3.4 \times 10^{-9}$` and `$\rho_{\mathrm{partial}} = -0.288$, $p = 5.5 \times 10^{-14}$` with the corrected values, and **regenerate the figure** from `t19_galactic_radius_distfix_plot.png` (the running median changes with corrected radii).

### 6. Summary table, lines 450 & 458
**Old (450):** `T10 & Mantel spatial test  & 655 & $r = -0.010$, $p = 0.60$           & Spatially independent \\`
**New:** `T10 & Mantel spatial test  & 655 & $r = 0.067$, $p = 0.014$          & Weak spatial correlation \\`

**Old (458):** `T19 & Galactic gradient    & 655 & $\rho = -0.228$, $p = 3.4 \times 10^{-9}$ & Outer disc more coherent \\`
**New:** `T19 & Galactic gradient    & 655 & $\rho = -0.091$, $p = 0.02$ & Outer disc weakly more coherent \\`

### 7. Conclusions, line 587
**Old:**
> `\item Chemical coherence is spatially independent (Mantel $r = -0.010$, $p = 0.60$), satisfying a necessary condition for chemical tagging. Chemically similar clusters are not preferentially spatial neighbours.`

**New:**
> `\item The residual spatial--chemical correlation is weak (Mantel $r = 0.067$, $p = 0.014$; $<0.5$ per cent of variance): chemically similar clusters are only marginally more likely to be spatial neighbours, so spatial position is a minor rather than dominant confound for chemical tagging.`

### 8. Conclusions, line 595
**Old:**
> `\item Coherence varies with Galactic environment: outer disc clusters are more coherent than inner disc clusters ($\rho = -0.228$, $p = 3.4 \times 10^{-9}$), a gradient that persists after controlling for metallicity ($\rho_{\mathrm{partial}} = -0.288$, $p = 5.5 \times 10^{-14}$).`

**New:**
> `\item Coherence varies weakly with Galactic environment: outer disc clusters are slightly more coherent than inner disc clusters ($\rho = -0.091$, $p = 0.02$), a gradient that ⟨persists / weakens⟩ after controlling for metallicity (⟨FILL partial⟩).`

---

## Also note (not T10/T19 but same root cause)
- The intro/discussion lines that lean on the spatial-independence result (e.g. line 102 "spatial independence ... of the chemical fingerprint", line 571 "the outer disc preserves chemical coherence more effectively") should be softened to match the weaker corrected gradient and the weak spatial correlation.
- **T16b** (dissolved-member 3D positions / Mahalanobis) and **T20c** also consumed `dist_cl`; verify whether their numbers shift once regenerated. (Re-run status reported separately.)
- The abstract's framing "15 empirical tests" and the chemical-tagging thesis are not invalidated — but the two environment/spatial tests must be reported at their true (weaker) strength.
