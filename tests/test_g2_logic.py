# -*- coding: utf-8 -*-
"""
Offline tests of the g2 run engine (qudi.logic.g2_logic.G2Run), the evaluation and the saving, on the
correlation counter dummy's simulation with a fake clock, a fake laser gate and a fake optimizer
(RUN-CONFOCAL-NV-G2-MODULE-STEP2-20261030). No qudi application, no hardware; files go to a
TemporaryDirectory.

    python -B -m unittest tests/test_g2_logic.py
"""

import os
import tempfile
import unittest

import numpy as np

from qudi.hardware.dummy.correlation_counter_dummy import CorrelationSim
from qudi.logic.g2_logic import G2Run, evaluate, save_g2
from qudi.util.g2_model import raw_floor


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Laser:
    """ confocal_scan_io's laser_on/laser_off/laser_is_on; refuses while `scanning` (like the real one). """

    def __init__(self, sim, on=False):
        self.sim = sim
        self.scanning = False
        self._on = on
        self.calls = []
        sim.set_excitation(on)

    @property
    def laser_is_on(self):
        return self._on

    def laser_on(self):
        self.calls.append('on')
        if not self.scanning:
            self._on = True
            self.sim.set_excitation(True)

    def laser_off(self):
        self.calls.append('off')
        if not self.scanning:
            self._on = False
            self.sim.set_excitation(False)


class Optimizer:
    def __init__(self):
        self.running = False
        self.started = 0
        self.allow = True

    def start(self):
        self.started += 1
        self.running = True

    def stop(self):
        self.running = False

    def busy(self):
        return self.running

    def allowed(self):
        return self.allow


class Busy:
    def __init__(self):
        self.modules = {}

    def __call__(self, exclude=frozenset()):
        return {k: v for k, v in self.modules.items() if k not in exclude}


class Engine(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.sim = CorrelationSim(rates=(40e3, 46e3), rho=0.69, seed=5, clock=self.clock)
        self.sim.configure((1, 2), 500, 2001)
        self.laser = Laser(self.sim)
        self.busy = Busy()
        self.opt = Optimizer()
        self.messages = []
        self.run = G2Run(self.sim, laser=self.laser, busy=self.busy, refocus=self.opt, clock=self.clock,
                         log=lambda level, msg: self.messages.append((level, msg)), run_time_s=60)

    def step(self, n=1, dt=1.0):
        for _ in range(n):
            self.clock.t += dt
            self.run.poll()

    def test_run_to_time_and_laser_restored(self):
        ok, _ = self.run.start()
        self.assertTrue(ok)
        self.assertTrue(self.laser.laser_is_on)
        self.step(59)
        self.assertEqual(self.run.phase, 'running')
        self.step(2)
        self.assertEqual(self.run.phase, 'finished')
        self.assertFalse(self.laser.laser_is_on)          # g2 lit it, so g2 switched it off
        self.assertFalse(self.sim.is_running)
        self.assertAlmostEqual(self.sim.get_data()['capture_s'], 60.0, delta=1.0)

    def test_operator_laser_stays_on(self):
        self.laser = Laser(self.sim, on=True)
        self.run.laser = self.laser
        self.run.start()
        self.step(61)
        self.assertEqual(self.run.phase, 'finished')
        self.assertTrue(self.laser.laser_is_on)
        self.assertNotIn('off', self.laser.calls)

    def test_refuses_while_other_measurement_runs(self):
        self.busy.modules['odmr_logic'] = 'locked'
        ok, msg = self.run.start()
        self.assertFalse(ok)
        self.assertIn('odmr_logic', msg)
        self.assertFalse(self.laser.laser_is_on)
        self.assertFalse(self.sim.is_running)

    def test_watchdog_aborts_when_measurement_starts(self):
        self.run.start()
        self.step(5)
        self.busy.modules['scanning_optimize_logic'] = 'locked'
        self.laser.scanning = True                         # the optimizer's scan refuses laser_off
        self.step(1)
        self.assertEqual(self.run.phase, 'aborted')
        self.assertIn('scanning_optimize_logic', self.run.reason)
        self.assertTrue(self.run.restore_pending)
        self.laser.scanning = False
        del self.busy.modules['scanning_optimize_logic']
        self.step(1)                                       # retried after the scan
        self.assertFalse(self.run.restore_pending)
        self.assertFalse(self.laser.laser_is_on)

    def test_watchdog_aborts_on_laser_loss(self):
        self.run.start()
        self.step(5)
        self.sim.set_excitation(False)                     # light gone without the gate knowing
        self.step(1)
        self.assertEqual(self.run.phase, 'running')        # one low poll is tolerated
        self.step(1)
        self.assertEqual(self.run.phase, 'aborted')
        self.assertIn('laser or spot lost', self.run.reason)

    def test_refocus_pauses_and_continues(self):
        self.run.refocus_enabled = True
        self.run.start()
        self.step(12)
        capture_before = self.sim.get_data()['capture_s']
        self.sim.rates = np.array([30e3, 34e3])           # 74 % of the start: drift out of focus
        self.step(10)
        self.assertEqual(self.run.phase, 'refocus')
        self.assertEqual(self.opt.started, 1)
        self.assertFalse(self.sim.is_running)
        self.busy.modules['scanning_optimize_logic'] = 'locked'    # caused by the refocus: not an abort
        self.busy.modules['scanning_probe_logic'] = 'locked'
        self.step(5)
        self.assertEqual(self.run.phase, 'refocus')
        self.assertTrue(self.laser.laser_is_on)
        self.busy.modules.clear()
        self.sim.rates = np.array([40e3, 46e3])
        self.opt.running = False
        self.step(1)
        self.assertEqual(self.run.phase, 'running')
        self.assertEqual(len(self.run.refocus_log), 1)
        self.step(3)
        self.assertGreater(self.sim.get_data()['capture_s'], capture_before)
        self.assertGreater(self.run.ref_rate, 80e3)

    def test_refocus_refused_with_periodic_autofocus(self):
        self.run.refocus_enabled = True
        self.opt.allow = False
        ok, msg = self.run.start()
        self.assertFalse(ok)
        self.assertIn('autofocus', msg)

    def test_refocus_that_never_starts_aborts(self):
        self.run.refocus_enabled = True
        self.opt.start = lambda: None                      # queued start lost
        self.run.start()
        self.step(12)
        self.sim.rates = np.array([30e3, 34e3])
        self.step(10)
        self.assertEqual(self.run.phase, 'refocus')
        self.step(6)
        self.assertEqual(self.run.phase, 'aborted')

    def test_stop_and_continue_accumulate(self):
        self.run.start()
        self.step(20)
        self.run.stop()
        self.assertEqual(self.run.phase, 'stopped')
        c1 = self.sim.get_data()['capture_s']
        self.clock.t += 100                                # paused time is not counted
        ok, _ = self.run.start(resume=True)
        self.assertTrue(ok)
        self.step(10)
        self.assertAlmostEqual(self.sim.get_data()['capture_s'], c1 + 10, delta=1.5)

    def test_next_run_keeps_and_releases_g2_laser(self):
        self.run.start()
        self.step(5)
        self.busy.modules['scanning_probe_logic'] = 'locked'
        self.laser.scanning = True
        self.step(1)
        self.assertEqual(self.run.phase, 'aborted')
        self.assertTrue(self.run.restore_pending)          # laser_off refused during the scan
        self.busy.modules.clear()
        self.laser.scanning = False
        self.laser._on = True                              # still on when the next run starts
        ok, _ = self.run.start()
        self.assertTrue(ok)
        self.assertFalse(self.run.restore_pending)
        self.step(61)
        self.assertEqual(self.run.phase, 'finished')
        self.assertFalse(self.laser.laser_is_on)           # g2 still owns it and switches it off

    def test_counter_error_at_start_restores_laser(self):
        def broken():
            raise RuntimeError('counter gone')
        self.sim.start = broken
        ok, msg = self.run.start()
        self.assertFalse(ok)
        self.assertIn('counter gone', msg)
        self.assertFalse(self.laser.laser_is_on)
        self.assertNotEqual(self.run.phase, 'running')

    def test_poll_error_aborts_and_restores_laser(self):
        self.run.start()
        self.step(3)
        def broken():
            raise RuntimeError('tagger lost')
        self.sim.get_count_rates = broken
        self.step(1)
        self.assertEqual(self.run.phase, 'aborted')
        self.assertIn('tagger lost', self.run.reason)
        self.assertFalse(self.laser.laser_is_on)

    def test_laser_off_error_still_ends_run(self):
        self.run.start()
        self.step(3)
        def broken():
            raise RuntimeError('pulse streamer gone')
        self.laser.laser_off = broken
        self.run.stop()
        self.assertEqual(self.run.phase, 'stopped')
        self.assertTrue(self.run.restore_pending)

    def _into_refocus(self):
        self.run.refocus_enabled = True
        self.run.start()
        self.step(12)
        self.sim.rates = np.array([30e3, 34e3])
        self.step(10)
        self.assertEqual(self.run.phase, 'refocus')
        self.step(1)                                       # the fake optimizer reports busy

    def test_other_measurement_during_refocus_aborts_and_stops_optimizer(self):
        self._into_refocus()
        self.busy.modules['odmr_logic'] = 'locked'
        self.step(1)
        self.assertEqual(self.run.phase, 'aborted')
        self.assertIn('odmr_logic', self.run.reason)
        self.assertFalse(self.opt.running)

    def test_operator_stop_during_refocus_stops_optimizer(self):
        self._into_refocus()
        self.run.stop()
        self.assertEqual(self.run.phase, 'stopped')
        self.assertFalse(self.opt.running)

    def test_refocus_count_is_capped(self):
        self.run.max_refocus = 2
        self.run.refocus_enabled = True
        self.run.start()
        self.step(12)
        for _ in range(3):
            self.sim.rates = self.sim.rates * 0.8          # keeps fading; each refocus fails
            for _ in range(15):
                self.step(1)
                if self.run.phase == 'refocus':
                    self.step(1)                           # the engine sees the optimize running
                    self.opt.running = False               # it finishes without effect
            if self.run.phase == 'aborted':
                break
        self.assertEqual(self.run.phase, 'aborted')
        self.assertIn('after 2 refocuses', self.run.reason)

    def test_reference_is_highest_rate_of_the_first_seconds(self):
        self.sim.rates = np.array([10e3, 10e3])            # laser still settling
        self.run.start()
        self.step(1)
        self.sim.rates = np.array([40e3, 46e3])
        self.step(4)
        self.assertGreater(self.run.ref_rate, 80e3)

    def test_reset_rates(self):
        self.run.start()
        self.step(5)
        self.run.reset_rates()
        self.assertEqual(self.run.rate_log, [])
        self.step(2)
        self.assertEqual(len(self.run.rate_log), 2)

    def test_no_laser_gate_logs(self):
        run = G2Run(self.sim, laser=None, clock=self.clock, log=lambda l, m: self.messages.append((l, m)))
        ok, _ = run.start()
        self.assertTrue(ok)
        self.assertTrue(any('operator' in m for _, m in self.messages))


class EvaluateAndSave(unittest.TestCase):
    def test_correction_and_files(self):
        clock = Clock()
        sim = CorrelationSim(rates=(40e3, 46e3), rho=0.69, seed=9, clock=clock)
        sim.configure((1, 2), 500, 8001)
        run = G2Run(sim, clock=clock, run_time_s=600)
        run.start()
        for _ in range(601):
            clock.t += 1.0
            run.poll()
        self.assertEqual(run.phase, 'finished')
        rates = run.rate_summary()
        self.assertAlmostEqual(rates['on_cps'], 86e3, delta=1e3)
        on = rates['on_cps']
        background = dict(cps=on * (1 - 0.69), err_cps=100.0)
        res = evaluate(sim.get_data(), rates, background)
        self.assertAlmostEqual(res['rho'], 0.69, delta=0.01)
        self.assertLess(abs(res['fit']['g2_0_tt'] - raw_floor(0.69)), 0.06)
        self.assertLess(res['g2c_dip_tt'], 0.3)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_g2(tmp, sim.get_data(), {'Status': run.phase}, res, tag='test')
            self.assertTrue(os.path.isfile(path))
            self.assertTrue(any(f.endswith('.png') for f in os.listdir(os.path.dirname(path))))
            rows = np.loadtxt(path)
            self.assertEqual(rows.shape, (8001, 3))
            with open(path, encoding='utf-8') as f:
                head = f.read(4000)
            self.assertIn('g2c_dip_tt', head)


if __name__ == '__main__':
    unittest.main()
