import unittest

from joint_gate_screen import (
    periapsis_speed,
    disk_outer_radius_au,
    circularization_energy_j,
    disk_binding_energy_j,
    type1_wave_timescale_yr,
    classify_cell,
)


class JointGateScreenTests(unittest.TestCase):
    def test_periapsis_speed_increases_with_focusing(self):
        v = periapsis_speed(1.0, 1.0, 1.0)
        self.assertGreater(v, 1.0)

    def test_disk_truncation_is_linear_in_periapsis(self):
        self.assertAlmostEqual(disk_outer_radius_au(3.0, 0.3), 0.9)

    def test_circularization_energy_scales_as_e_squared(self):
        e1 = circularization_energy_j(0.1)
        e2 = circularization_energy_j(0.2)
        self.assertAlmostEqual(e2 / e1, 4.0, places=10)

    def test_disk_binding_energy_positive_when_disk_extends_beyond_inner_edge(self):
        E = disk_binding_energy_j(r_out_au=1.0, f_sigma=0.1, p=1.0)
        self.assertGreater(E, 0.0)

    def test_disk_binding_energy_zero_when_disk_is_fully_truncated(self):
        E = disk_binding_energy_j(r_out_au=0.04, f_sigma=1.0, p=1.0)
        self.assertEqual(E, 0.0)

    def test_wave_timescale_decreases_with_disk_density(self):
        t_low = type1_wave_timescale_yr(f_sigma=0.01, p=1.0, h0=0.035, flaring=0.0)
        t_high = type1_wave_timescale_yr(f_sigma=1.0, p=1.0, h0=0.035, flaring=0.0)
        self.assertGreater(t_low, t_high)
        self.assertAlmostEqual(t_low / t_high, 100.0, places=8)

    def test_truncated_disk_fails_before_energy_or_damping(self):
        row = classify_cell(
            donor_mass_msun=1.0,
            a_ref_au=0.723332,
            q_over_ad=1.0,
            v_inf_kms=1.0,
            truncation_factor=0.3,
            f_sigma=1.0,
            p=1.0,
            h0=0.035,
            flaring=0.0,
            remaining_myr=3.0,
            e_i=0.05,
            i_i_deg=1.0,
        )
        self.assertEqual(row["label"], "DISK_TRUNCATED")
        self.assertFalse(row["disk_presence_gate"])

    def test_low_e_low_i_dense_long_lived_disk_can_screen_compatible_when_not_truncated(self):
        row = classify_cell(
            donor_mass_msun=1.0,
            a_ref_au=0.9,
            q_over_ad=4.0,
            v_inf_kms=0.3,
            truncation_factor=0.5,
            f_sigma=1.0,
            p=1.0,
            h0=0.035,
            flaring=0.0,
            remaining_myr=3.0,
            e_i=0.05,
            i_i_deg=1.0,
        )
        self.assertEqual(row["label"], "SCREEN_COMPATIBLE")
        self.assertTrue(row["energy_gate"])
        self.assertTrue(row["damping_gate"])

    def test_high_e_state_is_hydro_required(self):
        row = classify_cell(
            donor_mass_msun=1.0,
            a_ref_au=0.9,
            q_over_ad=4.0,
            v_inf_kms=0.3,
            truncation_factor=0.5,
            f_sigma=1.0,
            p=1.0,
            h0=0.035,
            flaring=0.0,
            remaining_myr=3.0,
            e_i=0.7,
            i_i_deg=1.0,
        )
        self.assertEqual(row["label"], "HYDRO_REQUIRED_OR_TOO_SLOW")
        self.assertTrue(row["hydro_required"])


class PilotGridTests(unittest.TestCase):
    def test_pilot_grid_has_preregistered_row_count(self):
        from joint_gate_screen import iter_pilot_grid
        self.assertEqual(sum(1 for _ in iter_pilot_grid()), 120960)


if __name__ == "__main__":
    unittest.main()
