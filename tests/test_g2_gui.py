# -*- coding: utf-8 -*-
"""
Offscreen smoke test of the g2 window (RUN-CONFOCAL-NV-G2-MODULE-STEP3-20261030): builds G2MainWindow and
feeds G2Gui's update slots with a status, simulated data and a real fit, without a qudi application
(no Connector, no logic). Checks that nothing raises and that the widgets show the values.

    python -B -m unittest tests/test_g2_gui.py
"""

import os
import types
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from PySide6 import QtWidgets

from qudi.gui.g2.g2_gui import G2Gui, G2MainWindow
from qudi.hardware.dummy.correlation_counter_dummy import CorrelationSim
from qudi.logic.g2_logic import evaluate


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def gui_stub():
    """ An object with G2Gui's state and its update methods bound, around a real window. """
    stub = types.SimpleNamespace(_mw=G2MainWindow(), _fit=None, _t0=0.0, _last_tau=None)
    for name in ('_apply_status', '_apply_data', '_apply_result', '_draw_fit', '_apply_saved'):
        setattr(stub, name, types.MethodType(getattr(G2Gui, name), stub))
    return stub


class Window(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_status_data_result(self):
        g = gui_stub()
        g._apply_status(dict(phase='running', message='started', last_rates=[40e3, 46e3], ref_rate=86e3,
                             run_time_s=600.0, refocus_enabled=False, refocus_possible=True,
                             bin_width_ps=500, n_bins=8001, laser_gate=False, laser_note='',
                             background=dict(cps=26e3, err_cps=50.0, source='measured', time='t')))
        self.assertIn('running', g._mw.status_label.text())
        self.assertIn('No laser gate', g._mw.warning_label.text())
        self.assertIn('86.0', g._mw.rates_label.text())
        self.assertFalse(g._mw.start_button.isEnabled())
        self.assertTrue(g._mw.stop_button.isEnabled())
        self.assertIn('+-2000 ns', g._mw.window_label.text())
        self.assertIn('26.00', g._mw.bg_label.text())

        clock = Clock()
        sim = CorrelationSim(seed=3, clock=clock)
        sim.configure((1, 2), 500, 8001)
        sim.start()
        clock.t += 600
        d = sim.get_data()
        g._apply_data(dict(tau_ns=d['tau_ps'] / 1e3, g_norm=d['g_norm'], counts=d['counts'],
                           capture_s=d['capture_s'], rates=[40e3, 46e3]))
        self.assertEqual(len(g._mw.full_data.getData()[0]), 8001)
        res = evaluate(d, dict(on_cps=86e3, on_start_cps=86e3, on_end_cps=86e3, seconds=600.0),
                       dict(cps=26e3, err_cps=50.0))
        g._apply_result(res)
        self.assertGreater(len(g._mw.zoom_fit.getData()[0]), 100)
        values = {g._mw.result_table.item(i, 0).text(): g._mw.result_table.item(i, 1).text()
                  for i in range(g._mw.result_table.rowCount())}
        self.assertNotEqual(values['raw g2(0), counter level'], '-')
        self.assertIn('+-', values['rho = S / (S + B)'])

        g._apply_status(dict(phase='aborted', reason='odmr_logic started during the g2 run',
                             restore_pending=True, laser_gate=True))
        self.assertIn('aborted', g._mw.warning_label.text())
        self.assertIn('Laser possibly still on', g._mw.warning_label.text())
        self.assertTrue(g._mw.start_button.isEnabled())
        g._apply_saved('C:/x/g2.dat')
        self.assertIn('g2.dat', g._mw.saved_label.text())

        g._apply_result(dict())                            # cleared result: no stale fit
        self.assertEqual(len(g._mw.zoom_fit.getData()[0] or []), 0)
        self.assertEqual(g._mw.result_table.item(0, 1).text(), '-')
        g._apply_result(dict(fit=None, rho=0.5, rho_err=0.01))   # no fit, background only
        self.assertIn('0.500', g._mw.result_table.item(3, 1).text())

        g._apply_status(dict(phase='finished', background_running=True, bin_width_ps=1000, n_bins=1001))
        self.assertTrue(g._mw.stop_button.isEnabled())     # Stop cancels the background
        self.assertFalse(g._mw.start_button.isEnabled())
        self.assertEqual(g._mw.bin_width.value(), 1000)
        g._mw.close()

    def test_safe_tag(self):
        from qudi.logic.g2_logic import safe_tag
        self.assertEqual(safe_tag('B3_-1_2'), 'B3_-1_2')
        self.assertEqual(safe_tag('a/b:c?'), 'a_b_c')
        self.assertEqual(safe_tag(''), '')


if __name__ == '__main__':
    unittest.main()
