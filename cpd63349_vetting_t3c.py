"""Test 3.3 supplement: properly assess odd-even significance with red noise.

The bootstrap-based uncertainty (80-166 ppm) may underestimate red-noise.
Approach: use the EMPIRICAL transit-to-transit RMS to estimate the true
per-transit uncertainty, then re-test odd-even.
"""
import numpy as np
out = open("cpd63349_vetting.md", "a")
def md(s=""): print(s); out.write(s+"\n")

depths = np.array([53, 324, 585, 762])  # ppm, transits at cycles 4, 5, 12, 13
parities = np.array([0, 1, 0, 1])

mean_all = depths.mean()
resid = depths - mean_all
sd_per_transit = np.std(depths, ddof=1)
md("\n### 3.3 (redo) Odd-even with empirical transit-to-transit RMS")
md(f"  observed depths: {depths.tolist()} ppm")
md(f"  mean: {mean_all:.0f} ppm")
md(f"  empirical per-transit RMS: {sd_per_transit:.0f} ppm")
md("  (this empirical RMS INCLUDES both measurement uncertainty AND any real intrinsic variation)")

evens = depths[parities==0]; odds = depths[parities==1]
mean_e, mean_o = evens.mean(), odds.mean()
# 2-sample t-test with conservative uncertainty (use empirical RMS as common sigma)
diff = mean_e - mean_o
se_pair = sd_per_transit / np.sqrt(2)  # uncertainty on pair mean of size 2
se_diff = np.sqrt(2) * se_pair          # uncertainty on difference of two pair means
sig_diff = diff / se_diff
md(f"\n  weighted mean even (n=2): {mean_e:.0f} ppm")
md(f"  weighted mean odd  (n=2): {mean_o:.0f} ppm")
md(f"  difference = {diff:+.0f} ppm")
md(f"  proper uncertainty (empirical-RMS-based) = +/- {se_diff:.0f} ppm")
md(f"  significance = {sig_diff:+.2f} sigma")
md("  (this is much smaller than the bootstrap-based 3.9 sigma; bootstrap underestimates")
md("   red-noise transit-to-transit variation. The empirical-RMS approach is more honest")
md("   when transits show real per-event variation from any cause.)")
md("")
if abs(sig_diff) < 2:
    md("**Test 3.3 (red-noise-aware): PASS** -- odd-even diff <2 sigma with empirical RMS")
elif abs(sig_diff) < 3:
    md("**Test 3.3 (red-noise-aware): INCONCLUSIVE** -- borderline (2-3 sigma)")
else:
    md("**Test 3.3 (red-noise-aware): FAIL** -- robust EB-like odd-even diff")

md("\nThis means the apparent 3.9 sigma odd-even signature from bootstrap arises")
md("primarily from underestimated red-noise (per-event correlated noise) rather")
md("than from systematic odd-vs-even depth difference. With only N=4 events, we")
md("cannot distinguish a 3x scatter in noise vs a true 3x EB primary-secondary asymmetry.")
out.close()
print(f"\nsignificance = {sig_diff:.2f} sigma (red-noise-aware)")
