# -*- coding: utf-8 -*-
"""
GUI for G2Logic (confocal_nv, 2026-10-30, RUN-CONFOCAL-NV-G2-MODULE-STEP3-20261030).

One window: run controls (start, continue, stop, clear, fit, save), run time and histogram settings,
the optional refocus, the background (measured off the spot or entered), live rates, the g2 histogram
over the full window and +-50 ns around the dip with the fit, and the fit / correction results.

Example config for copy-paste:

    g2_gui:
        module.Class: 'g2.g2_gui.G2Gui'
        connect:
            g2_logic: 'g2_logic'

Copyright (c) 2021, the qudi developers. See the AUTHORS.md file at the top-level directory of this
distribution and on <https://github.com/Ulm-IQO/qudi-iqo-modules/>

This file is part of qudi.

Qudi is free software: you can redistribute it and/or modify it under the terms of
the GNU Lesser General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

Qudi is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with qudi.
If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from qudi.core.connector import Connector
from qudi.core.module import GuiBase
from qudi.util.g2_model import model

ZOOM_NS = 50.0


class G2MainWindow(QtWidgets.QMainWindow):
    """ Plain widget window; no .ui file. """

    RESULT_ROWS = (('g2_0_tt', 'raw g2(0), counter level'), ('dip_bins_tt', 'raw dip bins, counter level'),
                   ('g2_0', 'raw g2(0), fit level'), ('rho', 'rho = S / (S + B)'),
                   ('raw_floor_single', 'single-emitter floor 1 - rho^2'),
                   ('g2c_dip_tt', 'corrected dip, counter level'), ('g2c_0', 'corrected g2(0), fit'),
                   ('t0_ns', 't0 (ns, cable delay B - A)'), ('t1_ns', 't1 (ns)'), ('t2_ns', 't2 (ns)'),
                   ('chi2_red', 'reduced chi2'), ('counts_at_dip', 'coincidences in the dip bins'))

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('qudi: g2')
        central = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(central)

        row = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton('Start')
        self.continue_button = QtWidgets.QPushButton('Continue')
        self.stop_button = QtWidgets.QPushButton('Stop')
        self.clear_button = QtWidgets.QPushButton('Clear')
        self.fit_button = QtWidgets.QPushButton('Fit')
        self.save_tag = QtWidgets.QLineEdit()
        self.save_tag.setPlaceholderText('save tag (e.g. the site)')
        self.save_button = QtWidgets.QPushButton('Save')
        for w in (self.start_button, self.continue_button, self.stop_button, self.clear_button,
                  self.fit_button, self.save_tag, self.save_button):
            row.addWidget(w)
        layout.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Run time (s)'))
        self.run_time = QtWidgets.QDoubleSpinBox()
        self.run_time.setRange(1, 86400)
        self.run_time.setDecimals(0)
        self.run_time.setKeyboardTracking(False)
        row.addWidget(self.run_time)
        row.addWidget(QtWidgets.QLabel('Bin (ps)'))
        self.bin_width = QtWidgets.QSpinBox()
        self.bin_width.setRange(1, 1000000)
        row.addWidget(self.bin_width)
        row.addWidget(QtWidgets.QLabel('Bins (odd)'))
        self.n_bins = QtWidgets.QSpinBox()
        self.n_bins.setRange(3, 100001)
        self.n_bins.setSingleStep(2)
        row.addWidget(self.n_bins)
        self.apply_hist_button = QtWidgets.QPushButton('Apply histogram')
        row.addWidget(self.apply_hist_button)
        self.window_label = QtWidgets.QLabel('')
        row.addWidget(self.window_label)
        row.addStretch(1)
        self.refocus_checkbox = QtWidgets.QCheckBox('Refocus when the rate drops')
        row.addWidget(self.refocus_checkbox)
        layout.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Background:'))
        self.bg_seconds = QtWidgets.QDoubleSpinBox()
        self.bg_seconds.setRange(1, 600)
        self.bg_seconds.setValue(10)
        self.bg_seconds.setSuffix(' s')
        row.addWidget(self.bg_seconds)
        self.bg_button = QtWidgets.QPushButton('Take background here')
        self.bg_button.setToolTip('Move the scanner cursor off the spot first (same z); the summed rate is '
                                  'measured where the cursor is.')
        row.addWidget(self.bg_button)
        row.addWidget(QtWidgets.QLabel('or enter (kc/s)'))
        self.bg_entry = QtWidgets.QDoubleSpinBox()
        self.bg_entry.setRange(0, 1e5)
        self.bg_entry.setDecimals(2)
        row.addWidget(self.bg_entry)
        self.bg_set_button = QtWidgets.QPushButton('Set')
        self.bg_clear_button = QtWidgets.QPushButton('Clear')
        row.addWidget(self.bg_set_button)
        row.addWidget(self.bg_clear_button)
        self.bg_label = QtWidgets.QLabel('none')
        row.addWidget(self.bg_label)
        row.addStretch(1)
        layout.addLayout(row)

        self.status_label = QtWidgets.QLabel('Status: idle')
        self.warning_label = QtWidgets.QLabel('')
        self.warning_label.setStyleSheet('color: #d62728; font-weight: bold;')
        self.warning_label.setWordWrap(True)
        self.rates_label = QtWidgets.QLabel('Rates: -')
        self.rule_label = QtWidgets.QLabel(
            'While g2 runs, do not start scans, optimize, ODMR or pulsed: the run is aborted (data kept).')
        for w in (self.status_label, self.warning_label, self.rates_label, self.rule_label):
            layout.addWidget(w)

        plots = QtWidgets.QHBoxLayout()
        self.full_plot = pg.PlotWidget()
        self.zoom_plot = pg.PlotWidget()
        for plot, title in ((self.full_plot, 'g2, full window'), (self.zoom_plot, f'g2, +-{ZOOM_NS:.0f} ns')):
            plot.setLabel('bottom', 'tau', units='ns')
            plot.setLabel('left', 'g2 (normalised)')
            plot.setTitle(title)
            plot.showGrid(x=True, y=True, alpha=0.3)
            plots.addWidget(plot, 1)
        self.full_data = self.full_plot.plot(pen=None, symbol='o', symbolSize=2, symbolPen=None,
                                             symbolBrush=(31, 119, 180))
        self.zoom_data = self.zoom_plot.plot(pen=pg.mkPen((31, 119, 180), width=1), symbol='o', symbolSize=4,
                                             symbolPen=None, symbolBrush=(31, 119, 180))
        self.full_fit = self.full_plot.plot(pen=pg.mkPen((214, 39, 40), width=2))
        self.zoom_fit = self.zoom_plot.plot(pen=pg.mkPen((214, 39, 40), width=2))
        layout.addLayout(plots, 1)

        self.result_table = QtWidgets.QTableWidget(len(self.RESULT_ROWS), 2)
        self.result_table.setHorizontalHeaderLabels(['quantity', 'value'])
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        for i, (_, text) in enumerate(self.RESULT_ROWS):
            self.result_table.setItem(i, 0, QtWidgets.QTableWidgetItem(text))
            self.result_table.setItem(i, 1, QtWidgets.QTableWidgetItem('-'))
        self.result_table.setMaximumHeight(320)
        self.saved_label = QtWidgets.QLabel('')
        self.saved_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.result_table)
        layout.addWidget(self.saved_label)

        self.setCentralWidget(central)
        self.resize(1200, 900)


class G2Gui(GuiBase):
    """ Qudi GUI module for G2Logic. """

    _g2_logic = Connector(name='g2_logic', interface='G2Logic')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mw = None
        self._fit = None
        self._t0 = 0.0
        self._last_tau = None

    def on_activate(self):
        logic = self._g2_logic()
        self._mw = G2MainWindow()
        mw = self._mw
        mw.start_button.clicked.connect(logic.start_run)
        mw.continue_button.clicked.connect(logic.continue_run)
        mw.stop_button.clicked.connect(logic.stop_run)
        mw.clear_button.clicked.connect(logic.clear_data)
        mw.fit_button.clicked.connect(logic.fit)
        mw.save_button.clicked.connect(self._save)
        mw.run_time.valueChanged.connect(self._run_time_changed)
        mw.apply_hist_button.clicked.connect(self._apply_histogram)
        mw.refocus_checkbox.toggled.connect(self._refocus_toggled)
        mw.bg_button.clicked.connect(self._take_background)
        mw.bg_set_button.clicked.connect(self._set_background)
        mw.bg_clear_button.clicked.connect(self._clear_background)
        queued = QtCore.Qt.ConnectionType.QueuedConnection
        logic.sigStatusChanged.connect(self._apply_status, queued)
        logic.sigDataUpdated.connect(self._apply_data, queued)
        logic.sigResultUpdated.connect(self._apply_result, queued)
        logic.sigSaved.connect(self._apply_saved, queued)
        status = logic.status
        mw.bin_width.setValue(int(status.get('bin_width_ps') or 500))
        mw.n_bins.setValue(int(status.get('n_bins') or 8001))
        self._apply_status(status)
        self._apply_result(logic.result)
        self.show()

    def on_deactivate(self):
        logic = self._g2_logic()
        for signal, slot in ((logic.sigStatusChanged, self._apply_status),
                             (logic.sigDataUpdated, self._apply_data),
                             (logic.sigResultUpdated, self._apply_result),
                             (logic.sigSaved, self._apply_saved)):
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        mw = self._mw
        for button in (mw.start_button, mw.continue_button, mw.stop_button, mw.clear_button, mw.fit_button,
                       mw.save_button, mw.apply_hist_button, mw.bg_button, mw.bg_set_button,
                       mw.bg_clear_button):
            button.clicked.disconnect()
        mw.run_time.valueChanged.disconnect()
        mw.refocus_checkbox.toggled.disconnect()
        mw.close()

    def show(self):
        self._mw.show()
        self._mw.raise_()
        self._mw.activateWindow()

    # ------------------------------------------------------------------ user actions
    def _save(self):
        self._g2_logic().save(self._mw.save_tag.text().strip())

    def _run_time_changed(self, value):
        self._g2_logic().set_settings(run_time=float(value))

    def _apply_histogram(self):
        n = int(self._mw.n_bins.value())
        if n % 2 == 0:
            n += 1
        self._g2_logic().set_settings(bin_width_ps=int(self._mw.bin_width.value()), n_bins=n)

    def _refocus_toggled(self, checked):
        self._g2_logic().set_settings(refocus_enabled=bool(checked))

    def _take_background(self):
        self._g2_logic().take_background(float(self._mw.bg_seconds.value()))

    def _set_background(self):
        self._g2_logic().set_settings(background_cps=1e3 * float(self._mw.bg_entry.value()))

    def _clear_background(self):
        self._g2_logic().set_settings(background_cps=None)

    # ------------------------------------------------------------------ updates from the logic
    def _apply_status(self, status):
        mw = self._mw
        phase = status.get('phase', 'idle')
        active = phase in ('running', 'refocus')
        background = bool(status.get('background_running'))
        message = status.get('message', '')
        mw.status_label.setText(f'Status: {phase}' + (f' - {message}' if message else '')
                                + (' | measuring background' if background else ''))
        warn = []
        if phase == 'aborted':
            warn.append(f"Run aborted: {status.get('reason', '')}")
        if status.get('restore_pending'):
            warn.append('Laser possibly still on: laser_off was refused (scan running?); g2 keeps retrying.')
        if not status.get('laser_gate', True):
            warn.append('No laser gate connected: switch the laser on yourself.')
        mw.warning_label.setText('  '.join(warn))
        note = status.get('laser_note')
        rates = status.get('last_rates')
        ref = status.get('ref_rate')
        text = 'Rates: '
        if rates:
            text += f'A {rates[0] / 1e3:.1f} kc/s, B {rates[1] / 1e3:.1f} kc/s, sum {sum(rates) / 1e3:.1f} kc/s'
        else:
            text += '-'
        if ref:
            text += f' | start reference {ref / 1e3:.1f} kc/s'
        if status.get('refocus_count'):
            text += f" | refocuses {status['refocus_count']}"
        if note:
            text += f' | {note}'
        mw.rates_label.setText(text)
        for w in (mw.start_button, mw.continue_button, mw.clear_button, mw.apply_hist_button, mw.bg_button,
                  mw.bg_set_button, mw.bg_clear_button):
            w.setEnabled(not active and not background)
        mw.clear_button.setEnabled(not background)
        mw.stop_button.setEnabled(active)
        mw.bin_width.setEnabled(not active)
        mw.n_bins.setEnabled(not active)
        if not mw.run_time.hasFocus():
            mw.run_time.blockSignals(True)
            mw.run_time.setValue(float(status.get('run_time_s', 600.0)))
            mw.run_time.blockSignals(False)
        box = mw.refocus_checkbox
        box.blockSignals(True)
        box.setChecked(bool(status.get('refocus_enabled')))
        box.setEnabled(bool(status.get('refocus_possible')) or box.isChecked())
        box.blockSignals(False)
        bw, n = status.get('bin_width_ps'), status.get('n_bins')
        if bw and n:
            mw.window_label.setText(f'window +-{bw * (n // 2) / 1e3:.0f} ns')
        bg = status.get('background')
        mw.bg_label.setText('none' if not bg else
                            f"{bg['cps'] / 1e3:.2f} +- {bg.get('err_cps', 0) / 1e3:.2f} kc/s "
                            f"({bg.get('source', '')}, {bg.get('time', '')})")

    def _apply_data(self, data):
        tau = np.asarray(data['tau_ns'], float)
        g = np.asarray(data['g_norm'], float)
        self._last_tau = tau
        self._mw.full_data.setData(tau, g)
        sel = np.abs(tau - self._t0) <= ZOOM_NS
        self._mw.zoom_data.setData(tau[sel], g[sel])
        self._draw_fit()
        cap = float(data.get('capture_s', 0.0))
        counts = int(np.sum(data['counts']))
        self._mw.full_plot.setTitle(f'g2, full window  ({cap:.0f} s, {counts} coincidences)')

    def _apply_result(self, result):
        fit = (result or {}).get('fit')
        self._fit = fit
        self._t0 = float(fit['t0_ns']) if fit else 0.0
        merged = dict(fit or {})
        merged.update({k: v for k, v in (result or {}).items() if k != 'fit'})
        for i, (key, _) in enumerate(G2MainWindow.RESULT_ROWS):
            value = merged.get(key)
            err = merged.get(key + '_err') or merged.get(key.replace('_ns', '_err_ns'))
            if value is None:
                text = '-'
            elif isinstance(value, (int, np.integer)):
                text = str(int(value))
            else:
                text = f'{value:.3f}' + (f' +- {err:.3f}' if err is not None else '')
            self._mw.result_table.item(i, 1).setText(text)
        self._draw_fit()

    def _draw_fit(self):
        mw = self._mw
        if not self._fit or self._last_tau is None:
            mw.full_fit.setData([], [])
            mw.zoom_fit.setData([], [])
            return
        f = self._fit
        tau = self._last_tau
        curve = model(tau, f['level'], 1 - f['g2_0'], f['a'], f['t1_ns'], f['t2_ns'], f['t0_ns'], f['sigma_ns'])
        mw.full_fit.setData(tau, curve)
        sel = np.abs(tau - self._t0) <= ZOOM_NS
        mw.zoom_fit.setData(tau[sel], curve[sel])

    def _apply_saved(self, path):
        self._mw.saved_label.setText(f'Saved: {path}')
