import math
import numpy as np

from coherent_capture_v3.venus_overlap_exchange import (
    KMS_TO_AUYR,
    hyperbola_elements,
    true_anomaly_for_radius,
    specific_energy,
    classify_binding,
    wilson_interval,
    is_cold_exchange,
)


def test_hyperbola_periapsis_identity():
    a, e = hyperbola_elements(q_au=2.0, v_inf_kms=0.3, m_total_msun=2.0)
    assert a < 0
    assert e > 1
    assert math.isclose(a * (1.0 - e), 2.0, rel_tol=1e-12, abs_tol=1e-12)


def test_true_anomaly_reconstructs_start_radius():
    a, e = hyperbola_elements(q_au=2.0, v_inf_kms=0.5, m_total_msun=2.0)
    f = true_anomaly_for_radius(a, e, 100.0, inbound=True)
    p = a * (1.0 - e * e)
    r = p / (1.0 + e * math.cos(f))
    assert f < 0
    assert math.isclose(r, 100.0, rel_tol=1e-10)


def test_specific_energy_bound_and_unbound_signs():
    mu = 4 * math.pi**2
    r = np.array([1.0, 0.0, 0.0])
    vc = 2 * math.pi
    assert specific_energy(r, np.array([0.0, vc, 0.0]), mu) < 0
    assert specific_energy(r, np.array([0.0, 2.0 * vc, 0.0]), mu) > 0


def test_classification_requires_bound_to_sun_and_unbound_from_donor():
    assert classify_binding(-1.0, +0.2) == "sun_exchange"
    assert classify_binding(-1.0, -0.2) == "double_bound"
    assert classify_binding(+0.1, -0.2) == "donor_bound"
    assert classify_binding(+0.1, +0.2) == "unbound"


def test_cold_exchange_joint_gate():
    assert is_cold_exchange("sun_exchange", True, 0.05, 3.4)
    assert not is_cold_exchange("sun_exchange", True, 0.051, 3.4)
    assert not is_cold_exchange("sun_exchange", True, 0.05, 3.41)
    assert not is_cold_exchange("sun_exchange", False, 0.01, 1.0)


def test_zero_event_wilson_upper_is_nonzero():
    lo, hi = wilson_interval(0, 6144)
    assert lo == 0.0
    assert 0.0 < hi < 0.001
