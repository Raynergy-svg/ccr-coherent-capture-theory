"""EB-screen module for SDE>=7 + on-target hits.

Seven sub-tests with thresholds frozen from the CPD-63 349 Test #3 vetting
plus the Gaia RUWE binarity check added 2026-05-30 after the
2M19321727 / 2M16513518 catches (Gaia-flagged binaries that passed all six
photometric tests):

  1. SDE(2P) / SDE(1P) -- is the detected period the harmonic of an EB at 2P?
        < 0.7  PASS  (1P is the real period; EB at 2P disfavoured)
       0.7-1.3 INCONC
        > 1.3  FAIL  (2P fits better; likely EB folded at half period)

  2. Odd-even per-transit depth difference (empirical-RMS-based sigma)
       |dd_oe| / sigma_pair < 2  PASS
                              < 3  INCONC
                              >= 3 FAIL

  3. Positive-depth fraction of predicted transits visible in coverage
       >= 0.7  PASS
       0.5-0.7 INCONC
        < 0.5  FAIL

  4. Duration vs central-transit duration for stellar density
       observed/central_b0 in [0.4, 1.1]  PASS  (consistent with b ~ 0-0.7)
       0.2 - 0.4   INCONC  (grazing-consistent; cannot rule in/out)
       1.1 - 1.4   INCONC
       < 0.2 or > 1.4   FAIL
     The duration is refit on the LCs with a wider trial grid before the test --
     the upstream BLS pipeline caps trial durations at 4.8h which makes the
     fit useless for long-period planets that have ~10h central durations.

  5. Implied companion radius from depth
       < 1.0 R_Jup  PASS
       1-2 R_Jup    INCONC (could be inflated planet or brown dwarf edge)
       >= 2 R_Jup   FAIL  (stellar)

  6. Secondary eclipse depth at phase 0.5
       <= 100 ppm AND not significant  PASS
       100-300 ppm                     INCONC
       > 300 ppm                       FAIL (likely EB)

  7. Gaia DR3 RUWE (unresolved-binary signature; Belokurov et al. 2020)
       < 1.4    PASS  (consistent with single star)
       1.4-2.0  INCONC (suspected binary)
       >= 2.0   FAIL  (Gaia astrometric fit fails as single star -- binary;
                       EB-blend is the dominant remaining FP mode)
     RUWE = NaN (query failure) -> INCONC, never FAIL, so a flaky network
     can't kill a real candidate.

  8. Gaia DR3 binary catalog membership (NSS + vari_eclipsing_binary)
     A direct hit in nss_two_body_orbit or vari_eclipsing_binary means
     Gaia has already cataloged this source as a binary system or EB.
       n_nss = 0  AND n_vari_eb = 0   PASS
       n_nss > 0  OR  n_vari_eb > 0   FAIL

  9. TESS half-orbital harmonic systematic (added 2026-05-30 after the
     empirical finding that 8/11 SURVIVED-pre-TLS candidates clustered within
     0.5% of n*6.83d; null 18.3%, binomial p = 1.2e-4).
     The TESS spacecraft passes through Earth's outer radiation belts twice
     per orbit (every 6.83d), causing scattered-light and momentum-dump
     residuals at the orbital and half-orbital cadence. PDC removes most of
     it, but the residual integrates coherently across many orbital cycles
     and shows up as a "real, on-target, period-stable" BLS signal that
     passes every other vetting test EXCEPT TLS' transit-shape constraint.
       |P/6.83d - round(P/6.83d)| / round(...) < 0.005   FAIL
                                          0.005-0.010    INCONC
                                                > 0.010  PASS
     Skip the test entirely (PASS) for P < 30d where the systematic doesn't
     have enough leverage to dominate over a real short-period transit.

Combined verdict (now scaled to 9 sub-tests):
   any FAIL                   -> EB_REJECT
   else >=5 PASS, <=2 INCONC  -> SURVIVED
   else                       -> FLAG_REVIEW
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from astropy.timeseries import BoxLeastSquares
from scipy.signal import medfilt

# Pre-registered thresholds (do not edit during use)
TH_SDE_RATIO_PASS = 0.7
TH_SDE_RATIO_FAIL = 1.3
TH_ODDEVEN_PASS   = 2.0
TH_ODDEVEN_FAIL   = 3.0
TH_POSFRAC_PASS   = 0.7
TH_POSFRAC_FAIL   = 0.5
TH_DUR_LO_PASS    = 0.4
TH_DUR_HI_PASS    = 1.1
TH_DUR_LO_FAIL    = 0.2     # below 0.2 of central = clearly impossible geometry
TH_DUR_HI_FAIL    = 1.4     # above 1.4 = inconsistent with stellar density
TH_RP_PASS_RJUP   = 1.0
TH_RP_FAIL_RJUP   = 2.0
TH_SEC_PASS_PPM   = 100
TH_SEC_FAIL_PPM   = 300
TH_RUWE_PASS      = 1.4     # Belokurov+2020: single-star fit OK below this
TH_RUWE_FAIL      = 2.0     # >=2 = unambiguous unresolved binary
TESS_HALF_ORBIT_D = 6.83    # TESS spacecraft half-orbital period (Earth-belt crossings)
TH_TESS_HARM_FAIL = 0.005   # 0.5% offset from integer multiple
TH_TESS_HARM_PASS = 0.010   # >1% off is safe
TH_TESS_HARM_MIN_P = 30.0   # skip the test for short-P (real USP planets are at integer 6.83d by coincidence)

R_JUP_EARTH = 11.21
R_SUN_AU    = 0.00465047

def detrend_concat(sectors):
    """Re-detrend per sector and concatenate. sectors=[(sec,t,f), ...]."""
    all_t, all_f = [], []
    for entry in sectors:
        if len(entry) == 4:
            s, t, f, fp = entry
        else:
            s, t, f = entry[:3]
        if len(t) < 30: continue
        dt = np.median(np.diff(t))
        if dt <= 0: dt = 0.02
        box = max(3, int(0.5/dt))
        if box % 2 == 0: box += 1
        trend = medfilt(f, box)
        fd = f / trend
        sig = 1.4826 * np.median(np.abs(fd - 1.0))
        keep = np.abs(fd - 1.0) < 5*sig
        all_t.append(t[keep]); all_f.append(fd[keep])
    if not all_t:
        return np.array([]), np.array([])
    return np.concatenate(all_t), np.concatenate(all_f)

def predict_transits(T, T0, P):
    """Return [(n, t)] list of in-coverage predicted transits."""
    if len(T) == 0: return []
    n_min = int(np.floor((T.min() - T0) / P))
    n_max = int(np.ceil((T.max()  - T0) / P))
    out = []
    for n in range(n_min, n_max + 1):
        tp = T0 + n * P
        if T.min() <= tp <= T.max():
            out.append((n, tp))
    return out

def sub1_sde_ratio(T, F, P_obs, dur_d):
    """Compare BLS SDE at P_obs vs at 2*P_obs."""
    if len(T) < 100:
        return dict(name="SDE(2P)/SDE(1P)", value=float("nan"), verdict="INCONC", note="n_pts<100")
    bls = BoxLeastSquares(T, F)
    durs = np.array([max(dur_d*0.7, 0.02), dur_d, dur_d*1.3])
    def sde_at(P_center, width_frac=0.10):
        lo = max(P_center * (1 - width_frac), 0.5)
        hi = P_center * (1 + width_frac)
        n_grid = 5000
        pers = np.linspace(lo, hi, n_grid)
        pg = bls.power(pers, durs)
        i = int(np.argmax(pg.power))
        near = np.abs((pg.period - pg.period[i]) / pg.period[i]) < 0.005
        if (~near).sum() < 100: return float("nan")
        mu = float(np.nanmean(pg.power[~near]))
        sd = float(np.nanstd (pg.power[~near]))
        return (float(pg.power[i]) - mu) / sd if sd > 0 else 0.0
    sde_1 = sde_at(P_obs)
    sde_2 = sde_at(2 * P_obs)
    if sde_1 <= 0 or not np.isfinite(sde_1) or not np.isfinite(sde_2):
        return dict(name="SDE(2P)/SDE(1P)", value=float("nan"), verdict="INCONC",
                    note=f"sde1={sde_1:.2f} sde2={sde_2:.2f}")
    ratio = sde_2 / sde_1
    if ratio < TH_SDE_RATIO_PASS:    verdict = "PASS"
    elif ratio > TH_SDE_RATIO_FAIL:  verdict = "FAIL"
    else:                            verdict = "INCONC"
    return dict(name="SDE(2P)/SDE(1P)", value=float(ratio), sde_1P=float(sde_1),
                sde_2P=float(sde_2), verdict=verdict)

def sub2_odd_even(T, F, P, T0, dur_d, predicted):
    """Empirical-RMS odd-even depth difference."""
    depths = []
    parities = []
    for n, tp in predicted:
        in_tr = (T >= tp - dur_d/2) & (T <= tp + dur_d/2)
        win   = (T >= tp - 0.5) & (T <= tp + 0.5)
        out_tr = win & ~in_tr
        if in_tr.sum() < 5 or out_tr.sum() < 20: continue
        d = 1.0 - np.median(F[in_tr]) / np.median(F[out_tr])
        depths.append(d)
        parities.append(n % 2)
    depths = np.array(depths); parities = np.array(parities)
    n_e = (parities == 0).sum(); n_o = (parities == 1).sum()
    if n_e < 2 or n_o < 2:
        return dict(name="odd-even", value=float("nan"), n_even=int(n_e), n_odd=int(n_o),
                    verdict="INCONC", note=f"n_even={n_e} n_odd={n_o}")
    mean_e = depths[parities == 0].mean()
    mean_o = depths[parities == 1].mean()
    sd_per_transit = np.std(depths, ddof=1)
    se_pair = sd_per_transit / np.sqrt(2)
    se_diff = np.sqrt(2) * se_pair
    diff = mean_e - mean_o
    sig = abs(diff) / se_diff if se_diff > 0 else float("inf")
    if sig < TH_ODDEVEN_PASS:    verdict = "PASS"
    elif sig > TH_ODDEVEN_FAIL:  verdict = "FAIL"
    else:                        verdict = "INCONC"
    return dict(name="odd-even", value=float(sig), mean_even_ppm=float(mean_e*1e6),
                mean_odd_ppm=float(mean_o*1e6), n_even=int(n_e), n_odd=int(n_o),
                empirical_rms_ppm=float(sd_per_transit*1e6), verdict=verdict)

def sub3_posfrac(T, F, dur_d, predicted):
    """Fraction of in-data transits with positive (transit-like) local depth."""
    if not predicted:
        return dict(name="positive-depth-frac", value=float("nan"), n_visible=0,
                    verdict="INCONC", note="no predicted transits")
    npos = 0; nvisible = 0
    for n, tp in predicted:
        in_tr = (T >= tp - dur_d/2) & (T <= tp + dur_d/2)
        win   = (T >= tp - 0.5) & (T <= tp + 0.5)
        out_tr = win & ~in_tr
        if in_tr.sum() < 5 or out_tr.sum() < 20: continue
        d = 1.0 - np.median(F[in_tr]) / np.median(F[out_tr])
        nvisible += 1
        if d > 0: npos += 1
    if nvisible == 0:
        return dict(name="positive-depth-frac", value=float("nan"), n_visible=0,
                    verdict="INCONC", note="no transits in coverage")
    frac = npos / nvisible
    if frac >= TH_POSFRAC_PASS:    verdict = "PASS"
    elif frac < TH_POSFRAC_FAIL:   verdict = "FAIL"
    else:                          verdict = "INCONC"
    return dict(name="positive-depth-frac", value=float(frac),
                n_positive=int(npos), n_visible=int(nvisible), verdict=verdict)

def refit_duration(T, F, P_obs, T0_obs, dur_guess):
    """Refit duration on the detrended LC with a wider trial grid.

    The upstream BLS pipeline searches durations up to 0.2 d (4.8 h).
    For long-period planets that's too restrictive -- a P=200 d planet
    around a sun-like dwarf has a central transit duration ~9 h. This
    refit grid spans 0.03 d (43 min) to 0.6 d (14.4 h).
    """
    if len(T) < 100: return float(dur_guess), float("nan")
    bls = BoxLeastSquares(T, F)
    durs = np.array([0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22,
                     0.28, 0.35, 0.45, 0.55])
    pers = np.linspace(P_obs*0.99, P_obs*1.01, 200)
    pg = bls.power(pers, durs)
    i = int(np.argmax(pg.power))
    dur_refit = float(pg.duration[i])
    power = float(pg.power[i])
    return dur_refit, power

def sub4_duration(dur_d, P_d, M_star, R_star):
    """Observed duration / b=0 central-transit duration."""
    if M_star <= 0 or R_star <= 0 or P_d <= 0:
        return dict(name="duration", value=float("nan"), verdict="INCONC", note="bad stellar params")
    a_AU = (((P_d/365.25)**2) * M_star) ** (1.0/3.0)
    Rs_AU = R_star * R_SUN_AU
    dur_central = (P_d / np.pi) * (Rs_AU / a_AU)   # days, b=0
    if dur_central <= 0:
        return dict(name="duration", value=float("nan"), verdict="INCONC", note="dur_central<=0")
    ratio = dur_d / dur_central
    if TH_DUR_LO_PASS <= ratio <= TH_DUR_HI_PASS:    verdict = "PASS"
    elif ratio < TH_DUR_LO_FAIL or ratio > TH_DUR_HI_FAIL: verdict = "FAIL"
    else:                                            verdict = "INCONC"
    return dict(name="duration", value=float(ratio),
                obs_h=float(dur_d*24), central_h=float(dur_central*24), verdict=verdict)

def sub5_companion_R(depth, R_star):
    """Implied R_companion from depth and R*."""
    if depth <= 0 or R_star <= 0:
        return dict(name="companion-R", value=float("nan"), verdict="INCONC", note="non-positive depth/R*")
    R_p_Rsun = np.sqrt(depth) * R_star
    R_p_Rjup = R_p_Rsun * 109.2 / R_JUP_EARTH
    if R_p_Rjup < TH_RP_PASS_RJUP:   verdict = "PASS"
    elif R_p_Rjup >= TH_RP_FAIL_RJUP: verdict = "FAIL"
    else:                            verdict = "INCONC"
    return dict(name="companion-R", value=float(R_p_Rjup), R_p_Rearth=float(R_p_Rjup*R_JUP_EARTH),
                verdict=verdict)

def sub7_gaia_ruwe(ruwe):
    """Gaia DR3 RUWE binarity check. ruwe is float or NaN."""
    import math
    if ruwe is None or (isinstance(ruwe, float) and math.isnan(ruwe)):
        return dict(name="gaia-ruwe", value=float("nan"), verdict="INCONC",
                    note="RUWE unavailable (no Gaia match or query failed)")
    try:
        ruwe = float(ruwe)
    except Exception:
        return dict(name="gaia-ruwe", value=float("nan"), verdict="INCONC",
                    note="RUWE not numeric")
    if ruwe < TH_RUWE_PASS:   verdict = "PASS"
    elif ruwe >= TH_RUWE_FAIL: verdict = "FAIL"
    else:                     verdict = "INCONC"
    return dict(name="gaia-ruwe", value=float(ruwe), verdict=verdict)

def sub9_tess_orbit_harmonic(P_d):
    """Flag periods that lie within 0.5% of n * TESS_HALF_ORBIT_D = n * 6.83d.

    Empirical: 8/11 SURVIVED-pre-TLS long-period candidates in the wider-net
    BLS sample landed within 0.5% of n*6.83d (null 18.3%, binomial p=1.2e-4).
    These passed centroid + EB-screen + FAP + block-sweep but failed TLS
    shape, consistent with PDC-residual scattered-light at the spacecraft
    half-orbital cadence integrating coherently across many cycles."""
    import math
    if P_d < TH_TESS_HARM_MIN_P:
        return dict(name="tess-orbit-harm", value=float("nan"), verdict="PASS",
                    note=f"P<{TH_TESS_HARM_MIN_P}d, test n/a (USP regime)")
    n = max(1, round(P_d / TESS_HALF_ORBIT_D))
    nearest = n * TESS_HALF_ORBIT_D
    frac_off = abs(P_d - nearest) / nearest
    if frac_off < TH_TESS_HARM_FAIL:
        verdict = "FAIL"
    elif frac_off < TH_TESS_HARM_PASS:
        verdict = "INCONC"
    else:
        verdict = "PASS"
    return dict(name="tess-orbit-harm", value=float(frac_off),
                n_harmonic=int(n), nearest_d=float(nearest), verdict=verdict)

def sub8_gaia_binary_catalog(n_nss, n_vari_eb, gaia_available=True):
    """Direct Gaia DR3 binary catalog membership: NSS orbits + vari_eb."""
    if not gaia_available or n_nss is None or n_vari_eb is None:
        return dict(name="gaia-binary-cat", value=float("nan"),
                    n_nss_orbit=None, n_vari_eb=None, verdict="INCONC",
                    note="Gaia catalog query unavailable")
    n_nss = int(n_nss); n_vari_eb = int(n_vari_eb)
    if n_nss == 0 and n_vari_eb == 0:
        return dict(name="gaia-binary-cat", value=0.0,
                    n_nss_orbit=n_nss, n_vari_eb=n_vari_eb, verdict="PASS")
    # >0 in either = Gaia has cataloged the source as a binary or EB
    return dict(name="gaia-binary-cat", value=float(n_nss + n_vari_eb),
                n_nss_orbit=n_nss, n_vari_eb=n_vari_eb, verdict="FAIL",
                note=("NSS orbit entries: " + str(n_nss) +
                      "; vari_eb entries: " + str(n_vari_eb)))

def sub6_secondary(T, F, P, T0, dur_d):
    """Depth at phase 0.5 (a secondary eclipse would be there for EBs)."""
    phase = ((T - T0)/P) % 1.0
    ph = np.where(phase > 0.5, phase - 1.0, phase)
    dur_phase = dur_d / P / 2
    if dur_phase <= 0:
        return dict(name="secondary", value=float("nan"), verdict="INCONC")
    sec_in = (np.abs(ph - 0.5) < dur_phase) | (np.abs(ph + 0.5) < dur_phase)
    sec_out = (np.abs(ph - 0.5) > 4*dur_phase) & (np.abs(ph + 0.5) > 4*dur_phase) & (np.abs(ph) > 4*dur_phase)
    if sec_in.sum() < 30 or sec_out.sum() < 200:
        return dict(name="secondary", value=float("nan"),
                    n_in=int(sec_in.sum()), n_out=int(sec_out.sum()),
                    verdict="INCONC", note="insufficient cadences")
    d_sec_ppm = (1.0 - np.median(F[sec_in]) / np.median(F[sec_out])) * 1e6
    if d_sec_ppm <= TH_SEC_PASS_PPM:    verdict = "PASS"
    elif d_sec_ppm > TH_SEC_FAIL_PPM:   verdict = "FAIL"
    else:                               verdict = "INCONC"
    return dict(name="secondary", value=float(d_sec_ppm), verdict=verdict)

def run_eb_screen(sectors, P, T0, dur_d, depth, R_star, M_star, Teff=None,
                  ra=None, dec=None, ruwe=None,
                  n_nss=None, n_vari_eb=None):
    """Run all eight EB sub-tests. sectors is the list returned by the BLS pipeline.

    For the new Gaia tests (7+8) pass either:
      ruwe=, n_nss=, n_vari_eb= : precomputed Gaia values (preferred)
      ra=..., dec=...           : queried live via gaia_lookup.get_gaia_full_check
                                   (one TAP round-trip covers both sub-tests).
    If neither supplied, both Gaia sub-tests return INCONC (not FAIL) so a
    flaky network cannot kill a real candidate.

    Returns dict with per-sub-test results and an overall verdict:
      SURVIVED      -- 0 FAIL, <=2 INCONC, >=5 PASS (out of 8)
      FLAG_REVIEW   -- 0 FAIL but too many INCONC
      EB_REJECT     -- any FAIL
    """
    T, F = detrend_concat(sectors)
    predicted = predict_transits(T, T0, P)
    # Refit duration on the LC with a wider trial grid -- the upstream
    # BLS pipeline caps at 4.8 h which is useless for long-P candidates.
    dur_refit, dur_power = refit_duration(T, F, P, T0, dur_d)
    dur_for_tests = dur_refit if dur_refit > 0 else dur_d

    # Resolve Gaia values: prefer explicit, else single live query covering 7+8.
    gaia_ok = True
    if ruwe is None and ra is not None and dec is not None:
        try:
            from gaia_lookup import get_gaia_full_check
            ruwe, _sid, _ncone, n_nss, n_vari_eb = get_gaia_full_check(ra, dec)
        except Exception:
            ruwe = float("nan"); n_nss = None; n_vari_eb = None
            gaia_ok = False

    s1 = sub1_sde_ratio(T, F, P, dur_for_tests)
    s2 = sub2_odd_even (T, F, P, T0, dur_for_tests, predicted)
    s3 = sub3_posfrac  (T, F, dur_for_tests, predicted)
    s4 = sub4_duration (dur_for_tests, P, M_star, R_star)
    s4["dur_input_d"] = float(dur_d)
    s4["dur_refit_d"] = float(dur_refit)
    s5 = sub5_companion_R(depth, R_star)
    s6 = sub6_secondary (T, F, P, T0, dur_for_tests)
    s7 = sub7_gaia_ruwe (ruwe)
    s8 = sub8_gaia_binary_catalog(n_nss, n_vari_eb, gaia_available=gaia_ok)
    s9 = sub9_tess_orbit_harmonic(P)
    subs = [s1, s2, s3, s4, s5, s6, s7, s8, s9]
    n_fail   = sum(1 for s in subs if s["verdict"] == "FAIL")
    n_inconc = sum(1 for s in subs if s["verdict"] == "INCONC")
    n_pass   = sum(1 for s in subs if s["verdict"] == "PASS")
    if n_fail >= 1:                    overall = "EB_REJECT"
    elif n_pass >= 5 and n_inconc <= 2: overall = "SURVIVED"
    else:                              overall = "FLAG_REVIEW"
    return dict(verdict=overall, n_pass=n_pass, n_inconc=n_inconc, n_fail=n_fail,
                n_predicted_transits=len(predicted), n_pts=len(T),
                sub_tests=subs)
