from __future__ import annotations

import hashlib
import itertools
import json
import math
from typing import Iterable

G = 6.67430e-11
AU = 1.495978707e11
YEAR_S = 365.25 * 86400.0
M_SUN = 1.98847e30
M_VENUS = 4.8675e24
A_VENUS_AU = 0.723332
SIGMA_1AU = 17000.0  # 1700 g cm^-2 -> kg m^-2
R_IN_AU = 0.05

DONOR_MASS_GRID = (0.50, 0.75, 1.00, 1.25)
A_REF_GRID = (0.60, 0.723332, 0.90)
Q_OVER_AD_GRID = (0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 3.00, 4.00)
VINF_GRID = (0.10, 0.30, 0.50, 1.00, 2.00, 3.00, 5.00, 10.00)
TRUNCATION_GRID = (0.2, 0.3, 0.5)
F_SIGMA_GRID = (0.01, 0.03, 0.10, 0.30, 1.00)
P_GRID = (0.5, 1.0, 1.5)
H0_GRID = (0.025, 0.035, 0.050)
FLARING_GRID = (0.0, 0.25)
REMAINING_MYR_GRID = (0.1, 0.3, 1.0, 3.0)
E_GRID = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90)
I_GRID_DEG = (1.0, 3.4, 5.0, 10.0, 20.0, 30.0, 60.0, 90.0)


def periapsis_speed(v_inf_kms: float, q_au: float, donor_mass_msun: float) -> float:
    q_m = q_au * AU
    vinf = v_inf_kms * 1000.0
    vp = math.sqrt(vinf * vinf + 2.0 * G * M_SUN * (1.0 + donor_mass_msun) / q_m)
    return vp / 1000.0


def disk_outer_radius_au(q_au: float, truncation_factor: float) -> float:
    return q_au * truncation_factor


def circularization_energy_j(e_i: float, a_au: float = A_VENUS_AU) -> float:
    return G * M_SUN * M_VENUS * e_i * e_i / (2.0 * a_au * AU)


def _dimensionless_power_integral(x0: float, x1: float, p: float) -> float:
    if x1 <= x0:
        return 0.0
    if math.isclose(p, 1.0, rel_tol=0.0, abs_tol=1e-14):
        return math.log(x1 / x0)
    return (x1 ** (1.0 - p) - x0 ** (1.0 - p)) / (1.0 - p)


def disk_binding_energy_j(r_out_au: float, f_sigma: float, p: float, r_in_au: float = R_IN_AU) -> float:
    if r_out_au <= r_in_au:
        return 0.0
    integral_sigma_dr = f_sigma * SIGMA_1AU * AU * _dimensionless_power_integral(r_in_au, r_out_au, p)
    return math.pi * G * M_SUN * integral_sigma_dr


def aspect_ratio(r_au: float, h0: float, flaring: float) -> float:
    return h0 * (r_au ** flaring)


def sigma_at_radius_kg_m2(r_au: float, f_sigma: float, p: float) -> float:
    return f_sigma * SIGMA_1AU * (r_au ** (-p))


def type1_wave_timescale_yr(
    f_sigma: float,
    p: float,
    h0: float,
    flaring: float,
    a_au: float = A_VENUS_AU,
) -> float:
    a_m = a_au * AU
    sigma = sigma_at_radius_kg_m2(a_au, f_sigma, p)
    h = aspect_ratio(a_au, h0, flaring)
    omega = math.sqrt(G * M_SUN / (a_m ** 3))
    t_s = (M_SUN / M_VENUS) * (M_SUN / (sigma * a_m * a_m)) * (h ** 4) / omega
    return t_s / YEAR_S


def _config_hash(inputs: dict) -> str:
    payload = json.dumps(inputs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def classify_cell(
    *,
    donor_mass_msun: float,
    a_ref_au: float,
    q_over_ad: float,
    v_inf_kms: float,
    truncation_factor: float,
    f_sigma: float,
    p: float,
    h0: float,
    flaring: float,
    remaining_myr: float,
    e_i: float,
    i_i_deg: float,
    donor_e: float = 0.0,
) -> dict:
    inputs = dict(
        donor_mass_msun=donor_mass_msun,
        a_ref_au=a_ref_au,
        q_over_ad=q_over_ad,
        v_inf_kms=v_inf_kms,
        truncation_factor=truncation_factor,
        f_sigma=f_sigma,
        p=p,
        h0=h0,
        flaring=flaring,
        remaining_myr=remaining_myr,
        e_i=e_i,
        i_i_deg=i_i_deg,
        donor_e=donor_e,
    )
    a_d_au = donor_mass_msun * a_ref_au
    q_au = q_over_ad * a_d_au
    r_out_au = disk_outer_radius_au(q_au, truncation_factor)
    v_p_kms = periapsis_speed(v_inf_kms, q_au, donor_mass_msun)
    delta_e_j = circularization_energy_j(e_i)
    e_disk_j = disk_binding_energy_j(r_out_au, f_sigma, p)

    disk_presence_gate = r_out_au >= A_VENUS_AU
    energy_gate = delta_e_j <= e_disk_j

    h = aspect_ratio(A_VENUS_AU, h0, flaring)
    e_over_h = e_i / h
    i_over_h = math.radians(i_i_deg) / h
    hydro_required = (e_over_h > 2.0) or (i_over_h > 2.0)

    t_wave_yr = type1_wave_timescale_yr(f_sigma, p, h0, flaring)
    t_e_yr = t_wave_yr / 0.780
    t_i_yr = t_wave_yr / 0.544
    remaining_yr = remaining_myr * 1e6
    damping_gate = (not hydro_required) and max(t_e_yr, t_i_yr) <= remaining_yr

    if not disk_presence_gate:
        label = "DISK_TRUNCATED"
    elif not energy_gate:
        label = "ENERGY_FAIL"
    elif hydro_required or not damping_gate:
        label = "HYDRO_REQUIRED_OR_TOO_SLOW"
    else:
        label = "SCREEN_COMPATIBLE"

    return {
        **inputs,
        "config_hash": _config_hash(inputs),
        "a_d_au": a_d_au,
        "q_au": q_au,
        "v_p_kms": v_p_kms,
        "r_out_au": r_out_au,
        "delta_e_j": delta_e_j,
        "disk_binding_energy_j": e_disk_j,
        "h_at_venus": h,
        "e_over_h": e_over_h,
        "i_over_h": i_over_h,
        "t_wave_yr": t_wave_yr,
        "t_e_yr": t_e_yr,
        "t_i_yr": t_i_yr,
        "disk_presence_gate": disk_presence_gate,
        "energy_gate": energy_gate,
        "hydro_required": hydro_required,
        "damping_gate": damping_gate,
        "label": label,
    }


def iter_locked_grid() -> Iterable[dict]:
    for values in itertools.product(
        DONOR_MASS_GRID,
        A_REF_GRID,
        Q_OVER_AD_GRID,
        VINF_GRID,
        TRUNCATION_GRID,
        F_SIGMA_GRID,
        P_GRID,
        H0_GRID,
        FLARING_GRID,
        REMAINING_MYR_GRID,
        E_GRID,
        I_GRID_DEG,
    ):
        yield classify_cell(
            donor_mass_msun=values[0], a_ref_au=values[1], q_over_ad=values[2],
            v_inf_kms=values[3], truncation_factor=values[4], f_sigma=values[5],
            p=values[6], h0=values[7], flaring=values[8], remaining_myr=values[9],
            e_i=values[10], i_i_deg=values[11],
        )


def iter_pilot_grid() -> Iterable[dict]:
    for values in itertools.product(
        Q_OVER_AD_GRID,
        (0.30, 1.00, 3.00),
        TRUNCATION_GRID,
        F_SIGMA_GRID,
        REMAINING_MYR_GRID,
        E_GRID,
        I_GRID_DEG,
    ):
        yield classify_cell(
            donor_mass_msun=1.0,
            a_ref_au=A_VENUS_AU,
            q_over_ad=values[0],
            v_inf_kms=values[1],
            truncation_factor=values[2],
            f_sigma=values[3],
            p=1.0,
            h0=0.035,
            flaring=0.25,
            remaining_myr=values[4],
            e_i=values[5],
            i_i_deg=values[6],
            donor_e=0.0,
        )
