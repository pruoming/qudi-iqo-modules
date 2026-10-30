# -*- coding: utf-8 -*-
"""
GUI for AutofocusPriorityLogic (confocal_nv, 2026-10-10, RUN-CONFOCAL-NV-AUTOFOCUS-20261010).

Periodic autofocus on/off, period, "Autofocus now", "Cancel", live status (phase, next run) and the refocus
history (shift per refocus, result, which measurements were paused).

Example config for copy-paste:

    autofocus_gui:
        module.Class: 'autofocus.autofocus_gui.AutofocusGui'
        connect:
            autofocus_logic: 'autofocus_logic'
"""

import time

from PySide6 import QtCore, QtWidgets

from qudi.core.connector import Connector
from qudi.core.module import GuiBase


class AutofocusMainWindow(QtWidgets.QMainWindow):
    """ Plain widget window; no .ui file. """

    COLUMNS = ('time', 'reason', 'result', 'dx_um', 'dy_um', 'dz_um', 'paused', 'duration_s', 'note')
    HEADERS = ('Time', 'Trigger', 'Result', 'dx (um)', 'dy (um)', 'dz (um)', 'Paused', 'Duration (s)', 'Note')

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('qudi: Autofocus')
        central = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(central)

        controls = QtWidgets.QHBoxLayout()
        self.periodic_checkbox = QtWidgets.QCheckBox('Periodic autofocus')
        controls.addWidget(self.periodic_checkbox)
        controls.addWidget(QtWidgets.QLabel('every'))
        self.period_spinbox = QtWidgets.QDoubleSpinBox()
        self.period_spinbox.setRange(0.2, 1440.0)
        self.period_spinbox.setDecimals(1)
        self.period_spinbox.setSingleStep(1.0)
        self.period_spinbox.setSuffix(' min')
        self.period_spinbox.setKeyboardTracking(False)
        self.period_spinbox.setMinimumWidth(110)
        controls.addWidget(self.period_spinbox)
        controls.addStretch(1)
        self.now_button = QtWidgets.QPushButton('Autofocus now')
        controls.addWidget(self.now_button)
        self.cancel_button = QtWidgets.QPushButton('Cancel')
        self.cancel_button.setToolTip('Cancel the waiting or running autofocus: back to the start position, '
                                      'paused measurements resume')
        self.cancel_button.setEnabled(False)
        controls.addWidget(self.cancel_button)
        layout.addLayout(controls)

        self.status_label = QtWidgets.QLabel('Status: idle')
        self.next_label = QtWidgets.QLabel('Next periodic autofocus: disabled')
        self.rule_label = QtWidgets.QLabel(
            'Priority: pauses pulsed and ODMR (and ODMR CW output), waits for confocal image scans, time series '
            'untouched. Refocus at the current position; laser must be enabled for scans. Keep the POI Manager '
            'Track off; the POI list follows only if the scanner is at the active POI.')
        self.rule_label.setWordWrap(True)
        for label in (self.status_label, self.next_label, self.rule_label):
            layout.addWidget(label)

        self.history_table = QtWidgets.QTableWidget(0, len(self.COLUMNS))
        self.history_table.setHorizontalHeaderLabels(self.HEADERS)
        self.history_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.history_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.history_table, 1)

        self.setCentralWidget(central)
        self.resize(900, 420)


class AutofocusGui(GuiBase):
    """ Qudi GUI module for AutofocusPriorityLogic. """

    _autofocus_logic = Connector(name='autofocus_logic', interface='AutofocusPriorityLogic')

    _HISTORY_ROWS = 100

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mw = None
        self._next_due = None
        self._enabled = False
        self._refresh_timer = None

    def on_activate(self):
        logic = self._autofocus_logic()
        self._mw = AutofocusMainWindow()
        status = logic.status()
        self._apply_status(status)
        self._apply_history(logic.history)

        self._mw.periodic_checkbox.toggled.connect(self._periodic_toggled)
        self._mw.period_spinbox.valueChanged.connect(self._period_changed)
        self._mw.now_button.clicked.connect(self._autofocus_now)
        self._mw.cancel_button.clicked.connect(self._cancel)
        logic.sigStatusChanged.connect(self._apply_status, QtCore.Qt.ConnectionType.QueuedConnection)
        logic.sigHistoryUpdated.connect(self._apply_history, QtCore.Qt.ConnectionType.QueuedConnection)

        self._refresh_timer = QtCore.QTimer()
        self._refresh_timer.setInterval(1000)
        self._refresh_timer.timeout.connect(self._refresh_countdown)
        self._refresh_timer.start()
        self.show()

    def on_deactivate(self):
        logic = self._autofocus_logic()
        if self._refresh_timer is not None:
            self._refresh_timer.stop()
            self._refresh_timer.timeout.disconnect()
        for signal, slot in ((logic.sigStatusChanged, self._apply_status),
                             (logic.sigHistoryUpdated, self._apply_history)):
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        self._mw.periodic_checkbox.toggled.disconnect()
        self._mw.period_spinbox.valueChanged.disconnect()
        self._mw.now_button.clicked.disconnect()
        self._mw.cancel_button.clicked.disconnect()
        self._mw.close()

    def show(self):
        self._mw.show()
        self._mw.raise_()
        self._mw.activateWindow()

    # ------------------------------------------------------------------ user actions
    def _periodic_toggled(self, checked):
        self._autofocus_logic().set_periodic_enabled(checked)

    def _period_changed(self, minutes):
        self._autofocus_logic().set_period(minutes * 60.0)

    def _autofocus_now(self):
        self._autofocus_logic().autofocus_now()

    def _cancel(self):
        self._autofocus_logic().cancel()

    # ------------------------------------------------------------------ updates from the logic
    def _apply_status(self, status):
        self._enabled = bool(status.get('enabled'))
        self._next_due = status.get('next_due')
        checkbox = self._mw.periodic_checkbox
        checkbox.blockSignals(True)
        checkbox.setChecked(self._enabled)
        checkbox.blockSignals(False)
        spinbox = self._mw.period_spinbox
        if not spinbox.hasFocus():   # never overwrite a value the operator is typing
            spinbox.blockSignals(True)
            spinbox.setValue(float(status.get('period', 600.0)) / 60.0)
            spinbox.blockSignals(False)
        running = bool(status.get('running'))
        self._mw.now_button.setEnabled(not running)
        self._mw.cancel_button.setEnabled(running)
        phase = status.get('phase', 'idle')
        message = status.get('message', '')
        self._mw.status_label.setText(f'Status: {phase}' + (f' - {message}' if message else ''))
        self._refresh_countdown()

    def _refresh_countdown(self):
        if not self._enabled:
            text = 'Next periodic autofocus: disabled'
        elif self._next_due is None:
            text = 'Next periodic autofocus: after the running autofocus'
        else:
            remaining = max(0.0, self._next_due - time.time())
            text = f'Next periodic autofocus in {int(remaining // 60):d} min {int(remaining % 60):02d} s'
        self._mw.next_label.setText(text)

    def _apply_history(self, history):
        table = self._mw.history_table
        rows = list(history)[-self._HISTORY_ROWS:][::-1]   # newest first
        table.setRowCount(len(rows))
        for row, entry in enumerate(rows):
            for col, key in enumerate(AutofocusMainWindow.COLUMNS):
                table.setItem(row, col, QtWidgets.QTableWidgetItem(str(entry.get(key, ''))))
        table.resizeColumnsToContents()
