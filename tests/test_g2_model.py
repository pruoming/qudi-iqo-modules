# -*- coding: utf-8 -*-
"""
Offline tests of the g2 model and fit (qudi.util.g2_model) and of the correlation counter dummy's
simulation (RUN-CONFOCAL-NV-G2-MODULE-STEP1-20261030). No qudi application, no hardware.

    python -B -m pytest tests/test_g2_model.py
"""

import unittest

import numpy as np

from qudi.util import g2_model as g2
from qudi.hardware.dummy.correlation_counter_dummy import CorrelationSim

BIN_PS = 500
N_BINS = 8001


def histogram(p, rho=1.0, a=0.4, t1=10.0, t2=150.0, t0=3.2, lam=600.0, seed=1):
    """ Counts and normalised g of a correlation with the three-level shape, background-diluted by rho. """
    t = (np.arange(N_BINS) - N_BINS // 2) * BIN_PS / 1e3
    nv = g2.model(t, 1.0, p, a, t1, t2, t0)
    true = 1 - rho ** 2 + rho ** 2 * nv
    counts = np.random.default_rng(seed).poisson(lam * true).astype(float)
    return t, counts / lam, counts


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class Physics(unittest.TestCase):
    def test_floor_and_correction(self):
        self.assertAlmostEqual(g2.raw_floor(0.6), 0.64)
        self.assertAlmostEqual(g2.raw_floor(0.6, 2), 0.82)
        gc, _ = g2.correct(0.64, 0.01, 0.6, 0.0)
        self.assertAlmostEqual(gc, 0.0, places=9)
        gc, _ = g2.correct(0.82, 0.01, 0.6, 0.0)
        self.assertAlmostEqual(gc, 0.5, places=9)

    def test_correction_error_grows_with_rho_error(self):
        _, e0 = g2.correct(0.7, 0.02, 0.55, 0.0)
        _, e1 = g2.correct(0.7, 0.02, 0.55, 0.05)
        self.assertAlmostEqual(e0, 0.02 / 0.55 ** 2, places=9)
        self.assertGreater(e1, e0)

    def test_fit_recovers_single_emitter_behind_background(self):
        rho = 0.55
        t, g, n = histogram(p=1.0, rho=rho)
        f = g2.fit_g2(t, g, n)
        self.assertIsNotNone(f)
        true = g2.raw_floor(rho)
        self.assertLess(abs(f['g2_0'] - true), 4 * f['g2_0_err'] + 0.02)
        self.assertLess(abs(f['t0_ns'] - 3.2), 0.5)
        gc, _ = g2.correct(f['g2_0'], f['g2_0_err'], rho, 0.0)
        self.assertLess(gc, 0.5)
        self.assertLess(abs(f['dip_bins'] - true), 0.08)

    def test_two_emitters_stay_above_half_after_correction(self):
        rho = 0.6
        t, g, n = histogram(p=0.5, rho=rho, seed=3)
        f = g2.fit_g2(t, g, n)
        gc, ge = g2.correct(f['g2_0'], f['g2_0_err'], rho, 0.0)
        self.assertLess(abs(gc - 0.5), 4 * ge + 0.03)

    def test_flat_histogram(self):
        t, g, n = histogram(p=0.0, seed=5)
        f = g2.fit_g2(t, g, n)
        self.assertGreater(f['g2_0'], 0.9)

    def test_too_few_counts(self):
        t, g, n = histogram(p=1.0, lam=0.001, seed=7)
        self.assertIsNone(g2.fit_g2(t, g, n))


class DummySimulation(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.sim = CorrelationSim(rates=(40e3, 46e3), rho=0.69, delay_ns=3.5, seed=11, clock=self.clock)
        self.sim.configure((1, 2), BIN_PS, N_BINS)

    def test_accumulates_only_while_running(self):
        self.clock.t += 100
        self.assertEqual(self.sim.get_data()['counts'].sum(), 0)
        self.sim.start()
        self.clock.t += 10
        self.sim.stop()
        self.clock.t += 100
        d = self.sim.get_data()
        self.assertAlmostEqual(d['capture_s'], 10.0)
        # the mean of g2_raw over +-2 us is above 1 (bunching shoulder), so take it from the model
        mean_g = float(np.mean(1 - 0.69 ** 2 + 0.69 ** 2 * g2.model(d['tau_ps'] / 1e3, 1.0, 1.0, 0.4, 7.0,
                                                                      800.0, 3.5)))
        expected = 40e3 * 46e3 * BIN_PS * 1e-12 * 10 * N_BINS * mean_g
        self.assertLess(abs(d['counts'].sum() / expected - 1), 0.01)

    def test_dip_position_and_depth(self):
        self.sim.start()
        self.clock.t += 600
        d = self.sim.get_data()
        f = g2.fit_g2(d['tau_ps'] / 1e3, d['g_norm'], d['counts'])
        self.assertIsNotNone(f)
        self.assertLess(abs(f['t0_ns'] - 3.5), 0.3)
        self.assertLess(abs(f['g2_0_tt'] - g2.raw_floor(0.69)), 0.06)
        self.assertLess(abs(f['t1_ns'] - 7.0), 1.5)

    def test_laser_off_flattens_and_drops_rates(self):
        self.sim.get_count_rates()
        self.sim.set_excitation(False)
        self.sim.start()
        self.clock.t += 20
        rates = self.sim.get_count_rates()
        self.assertLess(rates.sum(), 2000)
        # dark counts barely coincide: ~0.0005 per bin in 20 s instead of ~18 with the laser on
        self.assertLess(self.sim.get_data()['counts'].sum(), 50)

    def test_clear_and_configure_reset(self):
        self.sim.start()
        self.clock.t += 5
        self.sim.clear()
        self.assertEqual(self.sim.get_data()['counts'].sum(), 0)
        self.sim.stop()
        self.sim.configure((1, 2), 1000, 101)
        d = self.sim.get_data()
        self.assertEqual(d['counts'].size, 101)
        self.assertEqual(int(d['tau_ps'][0]), -50 * 1000)


if __name__ == '__main__':
    unittest.main()
