# -*- coding: utf-8 -*-
"""
Pulsed ODMR GUI (2026-10-23/24): the stock ODMR GUI (frequency scan, plots, fits, saving) on a
PulsedOdmrScanInput, laid out in pages like the pulsed measurement GUI. The page tabs sit in
their own row at the top of the window, under the toolbar:

* 'ODMR' page: the stock ODMR plot and docks, plus a 'Pulse Sequence' dock (about 1/3 of the
  window width) with the settings that define the pulse sequence itself (pi, readout laser
  length, gap, wait). They are applied between sweeps (refused while a scan runs).
* 'Pulse Extraction' page (full window; the ODMR docks hide while it is shown), laid out like the pulsed
  GUI's: readout/reference rate per frequency point (top left), 'Parameters for Pulse
  Extraction' with the analysis/reference window Start and Width (top right), and 'Pulses'
  (bottom) with Display raw, a point selector ('sum' or one frequency) and the windows as
  draggable lines. With software windows (PulsedOdmrScanInput
  software_windows + PulsedOdmrLogic) a window change applies immediately, also during a scan,
  and re-analyses all accumulated sweeps, like the analysis windows of the pulsed GUI. Without
  software windows the windows are hardware gates and apply between sweeps.

Windows are given relative to the LASER FLANK (pulsed GUI convention: the flank found by
gated_conv_deriv, gaussian 20 ns + gradient argmax, on the summed trace). The plot shows the
extracted pulse (0 = flank) or, with 'Display raw', Pulse Streamer time from the readout laser
gate edge.

Example config:

    pulsed_odmr_gui:
        module.Class: 'odmr.pulsed_odmrgui.PulsedOdmrGui'
        connect:
            odmr_logic: pulsed_odmr_logic
            pulsed_input: pulsed_odmr_scan_input
"""

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets
from scipy import ndimage

from qudi.core.connector import Connector
from qudi.core.statusvariable import StatusVar
from qudi.gui.odmr.odmrgui import OdmrGui
from qudi.util.colordefs import QudiPalettePale as palette
from qudi.util.widgets.scientific_spinbox import ScienDSpinBox

FLANK_CONV_STD_NS = 20.0     # pulsed GUI default conv_std_dev (20 bins of 1 ns)
FLANK_MIN_COUNTS = 2000      # below this the trace is too sparse to place a flank
FLANK_MIN_CONTRAST = 5.0     # lit / dark mean around the flank; less = no real laser pulse
SEQUENCE_DOCK_FRACTION = 0.32  # Pulse Sequence dock width / window width (the plots get the rest)

SEQUENCE_FIELDS = (('pi_pulse_ns', 'pi pulse'),
                   ('readout_ns', 'Readout laser'),
                   ('gap_ns', 'Gap (around pi)'),
                   ('wait_ns', 'Wait after readout'))
WINDOW_FIELDS = (('detect_delay_ns', 'Readout window start (after flank)'),
                 ('detect_ns', 'Readout window width'),
                 ('ref_delay_ns', 'Reference start (after flank)'),
                 ('ref_ns', 'Reference width'))
FLANK_FIELDS = ('detect_delay_ns', 'ref_delay_ns')
REF_FIELDS = ('ref_delay_ns', 'ref_ns')


def detect_flanks(t_ns, counts, conv_std_ns=FLANK_CONV_STD_NS):
    """ Rising/falling flank indices like pulsed gated_conv_deriv, or None without data. """
    counts = np.asarray(counts, dtype=float)
    if counts.size < 3 or counts.sum() <= 0:
        return None
    bw = float(t_ns[1] - t_ns[0])
    conv = ndimage.gaussian_filter1d(counts, conv_std_ns / bw)
    deriv = np.gradient(conv)
    if not deriv.any():
        return None
    rising, falling = sorted([int(deriv.argmax()), int(deriv.argmin())])
    return rising, falling


def flank_is_trustworthy(counts, flanks):
    """ Only a real laser pulse may set the flank used for the windows: enough counts, a
    dark stretch before the rising flank and a lit stretch after it (pulse at least ~100 ns
    long). A noisy or laser-off trace returns False, and the last good flank is kept. """
    if flanks is None:
        return False
    counts = np.asarray(counts, dtype=float)
    rising, falling = flanks
    if counts.sum() < FLANK_MIN_COUNTS or rising < 5 or falling - rising < 50:
        return False
    dark = counts[max(0, rising - 50):max(1, rising - 10)].mean()
    lit = counts[rising + 25:min(falling, rising + 275)].mean()
    return lit >= FLANK_MIN_CONTRAST * max(dark, 1e-9) and lit > 0


def _spinbox(minimum):
    box = QtWidgets.QSpinBox()
    box.setRange(minimum, 10_000_000)
    box.setSuffix(' ns')
    box.setKeyboardTracking(False)
    return box


def _time_spinbox(minimum_s):
    """ Window time field like the pulsed GUI's (seconds with SI prefix, e.g. '68.000 ns'). """
    box = ScienDSpinBox()
    box.setSuffix('s')
    box.setDecimals(3)
    box.setRange(minimum_s, 1e-3)
    box.setMinimumWidth(110)
    return box


class PulseSequenceWidget(QtWidgets.QWidget):
    """ Sequence settings (ODMR page): applied between sweeps. """

    sigApply = QtCore.Signal(dict)
    sigReload = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layout = QtWidgets.QFormLayout()
        self.spinboxes = {key: _spinbox(0) for key, _ in SEQUENCE_FIELDS}
        for key, label in SEQUENCE_FIELDS:
            layout.addRow(label, self.spinboxes[key])
        self.shots_label = QtWidgets.QLabel('-')
        self.shot_label = QtWidgets.QLabel('-')
        layout.addRow('Shots per point', self.shots_label)
        layout.addRow('Shot length', self.shot_label)
        buttons = QtWidgets.QHBoxLayout()
        self.apply_button = QtWidgets.QPushButton('Apply')
        self.reload_button = QtWidgets.QPushButton('Reload')
        buttons.addWidget(self.apply_button)
        buttons.addWidget(self.reload_button)
        layout.addRow(buttons)
        self.status_label = QtWidgets.QLabel('applied between sweeps')
        self.status_label.setWordWrap(True)
        layout.addRow(self.status_label)
        self.setLayout(layout)
        self.apply_button.clicked.connect(lambda: self.sigApply.emit(self.values()))
        self.reload_button.clicked.connect(self.sigReload.emit)

    def values(self):
        return {key: int(box.value()) for key, box in self.spinboxes.items()}

    def set_values(self, params):
        for key, box in self.spinboxes.items():
            if params.get(key) is not None:
                box.blockSignals(True)
                box.setValue(int(params[key]))
                box.blockSignals(False)
        self.shots_label.setText(str(params.get('shots_per_point', '-')))
        self.shot_label.setText(f"{params.get('shot_ns', '-')} ns")

    def set_status(self, text, error=False):
        self.status_label.setStyleSheet('color: #e05050;' if error else '')
        self.status_label.setText(text)


class AnalysisWindowsWidget(QtWidgets.QGroupBox):
    """ 'Parameters for Pulse Extraction' like the pulsed GUI: analysis (readout) and reference
    window Start / Width relative to the laser flank, extraction method and flank. A finished
    edit (Enter, focus change) or a dragged window line is emitted once as sigChanged. """

    sigChanged = QtCore.Signal(dict)

    def __init__(self, *args, has_reference=True, **kwargs):
        super().__init__('Parameters for Pulse Extraction', *args, **kwargs)
        self._has_reference = has_reference
        self._last_emitted = None
        grid = QtWidgets.QGridLayout()
        right = QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        center = QtCore.Qt.AlignmentFlag.AlignCenter
        self.analysis_method = QtWidgets.QComboBox()
        self.analysis_method.addItem('mean_norm (readout / reference)' if has_reference
                                     else 'mean (readout)')
        self.analysis_method.setEnabled(False)
        grid.addWidget(QtWidgets.QLabel('Analysis Method:'), 0, 0, right)
        grid.addWidget(self.analysis_method, 0, 1, 1, 2)
        grid.addWidget(QtWidgets.QLabel('Start'), 1, 1, center)
        grid.addWidget(QtWidgets.QLabel('Width'), 1, 2, center)
        self.spinboxes = dict()
        for row, (label, start_key, width_key) in enumerate(
                (('Analysis Window:', 'detect_delay_ns', 'detect_ns'),
                 ('Reference Window:', 'ref_delay_ns', 'ref_ns')), start=2):
            grid.addWidget(QtWidgets.QLabel(label), row, 0, right)
            for col, key in ((1, start_key), (2, width_key)):
                box = _time_spinbox(-10e-6 if key in FLANK_FIELDS else 1e-9)
                if key in REF_FIELDS and not has_reference:
                    box.setEnabled(False)
                box.editingFinished.connect(self._edited)
                self.spinboxes[key] = box
                grid.addWidget(box, row, col)
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        grid.addWidget(line, 4, 0, 1, 3)
        self.extraction_method = QtWidgets.QComboBox()
        self.extraction_method.addItem('conv_deriv')
        self.extraction_method.setEnabled(False)
        grid.addWidget(QtWidgets.QLabel('Extraction Method:'), 5, 0, right)
        grid.addWidget(self.extraction_method, 5, 1, 1, 2)
        grid.addWidget(QtWidgets.QLabel('conv_std_dev:'), 6, 0, right)
        grid.addWidget(QtWidgets.QLabel(f'{FLANK_CONV_STD_NS:.2f} ns'), 6, 1, 1, 2)
        self.flank_label = QtWidgets.QLabel('-')
        self.flank_label.setWordWrap(True)
        grid.addWidget(QtWidgets.QLabel('Laser flank:'), 7, 0, right)
        grid.addWidget(self.flank_label, 7, 1, 1, 2)
        self.status_label = QtWidgets.QLabel('')
        self.status_label.setWordWrap(True)
        grid.addWidget(self.status_label, 8, 0, 1, 3)
        grid.setRowStretch(9, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        self.setLayout(grid)

    def values(self):
        """ Field values in ns; window starts are flank-relative. """
        return {key: int(round(box.value() * 1e9)) for key, box in self.spinboxes.items()
                if self._has_reference or key not in REF_FIELDS}

    def _set_box(self, key, value_ns):
        box = self.spinboxes[key]
        box.blockSignals(True)
        box.setValue(float(value_ns) * 1e-9)
        box.blockSignals(False)

    def allow_resend(self):
        """ The last emitted values were refused: the same values may be sent again. """
        self._last_emitted = None

    def _edited(self):
        vals = self.values()
        if vals != self._last_emitted:
            self._last_emitted = vals
            self.sigChanged.emit(vals)

    def set_values(self, windows, flank_ns):
        """ Show windows given in Pulse Streamer time as flank-relative values (no signal). """
        for key in self.spinboxes:
            val = windows.get(key)
            if val is None:
                continue
            self._set_box(key, round(val - flank_ns) if key in FLANK_FIELDS else val)
        self._last_emitted = self.values()

    def set_window_fields(self, windows):
        """ Fill the fields from dragged window lines (flank-relative ns) and emit once. """
        for key, val in windows.items():
            box = self.spinboxes.get(key)
            if box is not None and box.isEnabled():
                self._set_box(key, int(round(val)))
        self._edited()

    def set_flank(self, flank_ns, detected):
        how = 'detected in the trace' if detected else 'last known, no trace data yet'
        self.flank_label.setText(f'{flank_ns:.0f} ns after the laser gate edge ({how})')

    def set_status(self, text, error=False):
        self.status_label.setStyleSheet('color: #e05050;' if error else '')
        self.status_label.setText(text)


class PointRatesWidget(QtWidgets.QGroupBox):
    """ Top-left plot of the page (the pulsed GUI shows its measuring error there): readout and
    reference rate per frequency point from the summed histograms with the current windows. """

    def __init__(self, *args, has_reference=True, **kwargs):
        super().__init__('Readout and Reference per Point', *args, **kwargs)
        layout = QtWidgets.QVBoxLayout()
        self.plot = pg.PlotWidget()
        self.plot.setLabel('bottom', 'frequency', units='Hz')
        self.plot.setLabel('left', 'count rate', units='c/s')
        self.plot.addLegend()
        self.readout_curve = self.plot.plot([], [], pen=palette.c1, name='readout window')
        self.reference_curve = self.plot.plot([], [], pen=palette.c4, name='reference window') \
            if has_reference else None
        layout.addWidget(self.plot)
        self.setLayout(layout)

    def set_rates(self, rates):
        if rates is None:
            self.readout_curve.setData([], [])
            if self.reference_curve is not None:
                self.reference_curve.setData([], [])
            return
        freqs, readout, reference = rates
        self.readout_curve.setData(np.asarray(freqs), np.asarray(readout))
        if self.reference_curve is not None:
            self.reference_curve.setData(np.asarray(freqs), np.asarray(reference))


class TimeTraceWidget(QtWidgets.QGroupBox):
    """ 'Pulses' plot like the pulsed GUI's Pulse Extraction: the extracted laser pulse (time 0 =
    flank) or, with Display raw, Pulse Streamer time from the laser gate edge; the windows as
    draggable lines (analysis c3, reference c4); 'sum' or one frequency point. Window positions
    are kept flank-relative. """

    sigWindowsDragged = QtCore.Signal(dict)
    sigReset = QtCore.Signal()
    sigPointChanged = QtCore.Signal(int)     # -1 = sum, else point index

    def __init__(self, *args, has_reference=True, **kwargs):
        super().__init__('Pulses', *args, **kwargs)
        self._has_reference = has_reference
        self._flank_ns = 0.0
        self._trace = None
        self._flanks = None
        self._windows = (None, None, None, None)
        self._point_labels = []
        layout = QtWidgets.QVBoxLayout()
        top = QtWidgets.QHBoxLayout()
        self.raw_checkbox = QtWidgets.QCheckBox('Display raw')
        self.point_combo = QtWidgets.QComboBox()
        self.point_combo.addItem('sum')
        self.point_combo.setMinimumContentsLength(18)
        top.addWidget(self.raw_checkbox)
        top.addStretch(1)
        top.addWidget(QtWidgets.QLabel('Laser Pulse number:'))
        top.addWidget(self.point_combo)
        self.plot = pg.PlotWidget()
        self.plot.setLabel('bottom', 'time', units='s')
        self.plot.setLabel('left', 'events', units='#')
        self.curve = self.plot.plot([], [], pen=palette.c1)
        self.lines = {
            'sig_start': pg.InfiniteLine(pos=0, pen={'color': palette.c3, 'width': 1}, movable=True),
            'sig_end': pg.InfiniteLine(pos=0, pen={'color': palette.c3, 'width': 1}, movable=True),
            'ref_start': pg.InfiniteLine(pos=0, pen={'color': palette.c4, 'width': 1}, movable=True),
            'ref_end': pg.InfiniteLine(pos=0, pen={'color': palette.c4, 'width': 1}, movable=True)}
        for name, line in self.lines.items():
            if name.startswith('ref') and not has_reference:
                continue
            self.plot.addItem(line)
            line.sigPositionChangeFinished.connect(self._lines_moved)
        bottom = QtWidgets.QHBoxLayout()
        self.info_label = QtWidgets.QLabel('no data yet')
        self.reset_button = QtWidgets.QPushButton('Reset trace')
        bottom.addWidget(self.info_label, 1)
        bottom.addWidget(self.reset_button)
        layout.addLayout(top)
        layout.addWidget(self.plot, 1)
        layout.addLayout(bottom)
        self.setLayout(layout)
        self.reset_button.clicked.connect(self.sigReset.emit)
        self.raw_checkbox.stateChanged.connect(self._redraw)
        self.point_combo.currentIndexChanged.connect(
            lambda index: self.sigPointChanged.emit(index - 1))

    @property
    def show_raw(self):
        return self.raw_checkbox.isChecked()

    @property
    def point_index(self):
        """ -1 for the sum over all points, else the selected frequency point. """
        return self.point_combo.currentIndex() - 1

    def set_points(self, frequencies):
        """ Offer 'sum' plus one entry per frequency point (kept selection if still valid). """
        freqs = [] if frequencies is None else list(frequencies)
        labels = [f'{i}: {f * 1e-9:.4f} GHz' for i, f in enumerate(freqs)]
        if labels == self._point_labels:
            return
        keep = self.point_index
        self.point_combo.blockSignals(True)
        self.point_combo.clear()
        self.point_combo.addItem('sum')
        self.point_combo.addItems(labels)
        self.point_combo.setCurrentIndex(keep + 1 if keep < len(freqs) else 0)
        self.point_combo.blockSignals(False)
        self._point_labels = labels

    def _offset_ns(self):
        """ Plot x of a flank-relative time t is t + offset. """
        return self._flank_ns if (self.show_raw or self._flanks is None) else 0.0

    def set_trace(self, trace, flanks, flank_ns):
        self._trace, self._flanks, self._flank_ns = trace, flanks, float(flank_ns)
        self._redraw()

    def _redraw(self, *_):
        tr = self._trace
        if tr is not None:
            t = np.asarray(tr['t_ns'], dtype=float)
            counts = np.asarray(tr['counts'], dtype=float)
            if self.show_raw or self._flanks is None:
                x, y = t, counts
            else:
                r, f = self._flanks
                x, y = t[r:f] - t[r], counts[r:f]
            self.curve.setData(x * 1e-9, y)
            what = tr.get('label', 'summed over all points')
            self.info_label.setText(f"{int(counts.sum())} counts {what}, {tr['binwidth_ns']} ns "
                                    f"bins; {'raw (laser gate edge = 0)' if self.show_raw else 'extracted (flank = 0)'}")
        self._draw_windows()

    def set_windows(self, detect_start, detect_width, ref_start=None, ref_width=None):
        """ Flank-relative window geometry in ns. """
        self._windows = (detect_start, detect_width, ref_start, ref_width)
        self._draw_windows()

    def _draw_windows(self):
        off = self._offset_ns()
        d0, dw, r0, rw = self._windows
        for start_name, end_name, start, width in (('sig_start', 'sig_end', d0, dw),
                                                   ('ref_start', 'ref_end', r0, rw)):
            if start is None or width is None:
                continue
            for name, pos in ((start_name, start), (end_name, start + width)):
                line = self.lines[name]
                line.blockSignals(True)
                line.setValue((pos + off) * 1e-9)
                line.blockSignals(False)

    def _lines_moved(self, *_):
        """ Like the pulsed GUI: each pair gives start = smaller, end = larger position. """
        off = self._offset_ns()
        pos = {name: line.value() * 1e9 - off for name, line in self.lines.items()}
        a, b = sorted((pos['sig_start'], pos['sig_end']))
        windows = {'detect_delay_ns': a, 'detect_ns': max(b - a, 1.0)}
        if self._has_reference:
            c, d = sorted((pos['ref_start'], pos['ref_end']))
            windows.update(ref_delay_ns=c, ref_ns=max(d - c, 1.0))
        self._windows = (windows['detect_delay_ns'], windows['detect_ns'],
                         windows.get('ref_delay_ns'), windows.get('ref_ns'))
        self.sigWindowsDragged.emit(windows)


class PulseExtractionPage(QtWidgets.QWidget):
    """ 'Pulse Extraction' tab laid out like the pulsed GUI: per-point rates (top left),
    Parameters for Pulse Extraction (top right), Pulses (bottom, full width). """

    def __init__(self, *args, has_reference=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.rates = PointRatesWidget(has_reference=has_reference)
        self.windows = AnalysisWindowsWidget(has_reference=has_reference)
        self.trace = TimeTraceWidget(has_reference=has_reference)
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.rates, 0, 0)
        layout.addWidget(self.windows, 0, 1)
        layout.addWidget(self.trace, 1, 0, 1, 2)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(0, 2)
        layout.setRowStretch(1, 3)
        self.setLayout(layout)
        self.trace.sigWindowsDragged.connect(self.windows.set_window_fields)


class PulsedOdmrGui(OdmrGui):
    """ ODMR GUI for PulsedOdmrScanInput: 'ODMR' (with Pulse Sequence dock) and 'Pulse
    Extraction' (live analysis windows + time trace) tabs.

    Example config:

        pulsed_odmr_gui:
            module.Class: 'odmr.pulsed_odmrgui.PulsedOdmrGui'
            connect:
                odmr_logic: pulsed_odmr_logic
                pulsed_input: pulsed_odmr_scan_input
    """

    _pulsed_input = Connector(name='pulsed_input', interface='PulsedOdmrScanInput')
    # laser flank (ns after the readout laser gate edge) used until a trace is available;
    # 74 ns = the gated_conv_deriv rule on the first pulsed ODMR trace (2026-10-23). The pulsed
    # GUI's Rabi trace gives 54 ns: its turn-on spikes (absent in pulsed ODMR) move the flank.
    _last_flank_ns = StatusVar(name='last_flank_ns', default=74.0)

    sigSetAnalysisWindows = QtCore.Signal(dict)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tabs = None            # page switch (QTabBar in a top toolbar row)
        self._stack = None           # central pages: stock ODMR plot / Pulse Extraction
        self._page_toolbar = None
        self._hidden_docks = []      # ODMR docks hidden while Pulse Extraction is shown
        self._extraction_page = None
        self._window_panel = None
        self._trace_panel = None
        self._rates_panel = None
        self._sequence_panel = None
        self._sequence_dock = None
        self._trace_timer = None
        self._flank_detected = False
        self._loaded_windows = None
        self._live = False

    def on_activate(self):
        super().on_activate()
        self._mw.setWindowTitle('qudi: Pulsed ODMR')
        windows = self._pulsed_input().analysis_windows \
            if hasattr(self._pulsed_input(), 'analysis_windows') else {}
        self._live = bool(windows.get('software')) and hasattr(self._odmr_logic(),
                                                               'set_analysis_windows')
        self._build_tabs(bool(self._pulsed_input().pulse_params.get('ref_len_ns')))
        self._build_sequence_dock()
        self._sequence_panel.sigApply.connect(self._apply_sequence)
        self._sequence_panel.sigReload.connect(self._load_params)
        self._window_panel.sigChanged.connect(self._windows_edited)
        self._trace_panel.sigReset.connect(self._reset_trace)
        self._trace_panel.sigPointChanged.connect(lambda _i: self._refresh())
        if self._live:
            logic = self._odmr_logic()
            self.sigSetAnalysisWindows.connect(logic.set_analysis_windows,
                                               QtCore.Qt.ConnectionType.QueuedConnection)
            logic.sigAnalysisWindowsUpdated.connect(self._windows_applied,
                                                    QtCore.Qt.ConnectionType.QueuedConnection)
        self._load_params()
        self._place_sequence_dock()
        self._trace_timer = QtCore.QTimer()
        self._trace_timer.setInterval(1000)
        self._trace_timer.timeout.connect(self._refresh)
        self._trace_timer.start()
        self._refresh()

    def on_deactivate(self):
        if self._trace_timer is not None:
            self._trace_timer.stop()
            self._trace_timer.timeout.disconnect()
            self._trace_timer = None
        if self._live:
            try:
                self.sigSetAnalysisWindows.disconnect()
                self._odmr_logic().sigAnalysisWindowsUpdated.disconnect(self._windows_applied)
            except (RuntimeError, TypeError):
                pass
        # leave on the ODMR page, so the next start opens there
        if self._tabs is not None:
            self._tabs.setCurrentIndex(0)
        super().on_deactivate()
        self._tabs = self._extraction_page = self._window_panel = self._trace_panel = None
        self._rates_panel = self._sequence_panel = self._sequence_dock = None
        self._stack = self._page_toolbar = None
        self._hidden_docks = []

    # -- layout --------------------------------------------------------------------------

    def _build_tabs(self, has_reference):
        """ Pages like the pulsed GUI: an 'ODMR' / 'Pulse Extraction' tab bar in its own row at
        the top of the window (under the toolbar, above the docks) switches the central area
        between the stock plot and the Pulse Extraction page. The base class keeps its own
        reference to the plot widget (self._plot_widget), so it can leave the central slot
        (takeCentralWidget does not delete it). """
        plot = self._mw.takeCentralWidget()
        self._extraction_page = PulseExtractionPage(has_reference=has_reference)
        self._window_panel = self._extraction_page.windows
        self._trace_panel = self._extraction_page.trace
        self._rates_panel = self._extraction_page.rates
        self._stack = QtWidgets.QStackedWidget()
        self._stack.addWidget(plot)
        self._stack.addWidget(self._extraction_page)
        self._mw.setCentralWidget(self._stack)
        self._tabs = QtWidgets.QTabBar()
        self._tabs.addTab('ODMR')
        self._tabs.addTab('Pulse Extraction')
        self._tabs.setExpanding(False)
        self._page_toolbar = QtWidgets.QToolBar('Pages', self._mw)
        self._page_toolbar.setObjectName('pulsed_odmr_pages_toolbar')
        self._page_toolbar.setMovable(False)
        self._page_toolbar.setFloatable(False)
        self._page_toolbar.toggleViewAction().setVisible(False)    # the switch cannot be hidden
        self._page_toolbar.addWidget(self._tabs)
        self._mw.addToolBarBreak(QtCore.Qt.ToolBarArea.TopToolBarArea)
        self._mw.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, self._page_toolbar)
        self._tabs.currentChanged.connect(self._tab_changed)

    def _build_sequence_dock(self):
        self._sequence_panel = PulseSequenceWidget()
        self._sequence_dock = QtWidgets.QDockWidget('Pulse Sequence', self._mw)
        self._sequence_dock.setObjectName('pulse_sequence_dockwidget')
        self._sequence_dock.setWidget(self._sequence_panel)

    def _place_sequence_dock(self):
        if self._sequence_dock is None:
            return     # called from the base on_activate before the dock exists
        self._sequence_dock.setFloating(False)
        self._sequence_dock.show()
        self._mw.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self._sequence_dock)
        # Qt otherwise sizes the dock from its contents and can leave the plots only a third
        # of the window (bench 2026-10-24): give the dock ~1/3 once the window is laid out
        QtCore.QTimer.singleShot(0, self._size_sequence_dock)

    def _size_sequence_dock(self):
        dock = self._sequence_dock
        if dock is None or dock.isFloating() or not dock.isVisible():
            return
        width = max(dock.minimumSizeHint().width(), int(SEQUENCE_DOCK_FRACTION * self._mw.width()))
        self._mw.resizeDocks([dock], [width], QtCore.Qt.Orientation.Horizontal)

    def show(self):
        """ The base show restores the saved window state; a state saved before the page tab
        bar existed (or one Qt merged into the main toolbar row) is put right afterwards. A
        window re-shown on the Pulse Extraction page keeps the ODMR docks hidden. """
        super().show()
        self._fix_page_toolbar()
        if self._tabs is not None and self._tabs.currentIndex() == 1:
            self._tab_changed(1)

    def _fix_page_toolbar(self):
        """ Page tab bar: own row under the main toolbar, visible. """
        bar = self._page_toolbar
        if bar is None:
            return
        if self._mw.toolBarArea(bar) != QtCore.Qt.ToolBarArea.TopToolBarArea:
            self._mw.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, bar)
        if not self._mw.toolBarBreak(bar):
            self._mw.insertToolBarBreak(bar)
        bar.show()

    @QtCore.Slot()
    def restore_default_view(self):
        if self._tabs is not None:
            self._tabs.setCurrentIndex(0)
        super().restore_default_view()
        self._fix_page_toolbar()
        self._place_sequence_dock()

    def _odmr_docks(self):
        return [d for d in (self._scan_control_dockwidget, self._cw_control_dockwidget,
                            self._fit_dockwidget, self._sequence_dock) if d is not None]

    @QtCore.Slot(int)
    def _tab_changed(self, index):
        """ Pulse Extraction is a full page (P1, 2026-10-24, like the pulsed GUI): the ODMR docks
        (CW/Scan Control, Fit, Pulse Sequence) hide while it is shown and the ones that were
        visible come back on the ODMR page. """
        if self._stack is not None:
            self._stack.setCurrentIndex(index)
        if index == 1:
            if not self._hidden_docks:
                self._hidden_docks = [d for d in self._odmr_docks() if d.isVisible()]
            for dock in self._hidden_docks:
                dock.hide()
            self._refresh()
        else:
            for dock in self._hidden_docks:
                dock.show()
            self._hidden_docks = []

    # -- parameters ----------------------------------------------------------------------

    def _scan_running(self):
        return self._odmr_logic().module_state() != 'idle'

    def _current_windows(self):
        inp = self._pulsed_input()
        if hasattr(inp, 'analysis_windows'):
            return dict(inp.analysis_windows)
        p = inp.pulse_params
        return dict(detect_delay_ns=p['detect_delay_ns'], detect_ns=p['detect_len_ns'],
                    ref_delay_ns=p.get('ref_delay_ns'), ref_ns=p.get('ref_len_ns'))

    @QtCore.Slot()
    def _load_params(self):
        self._sequence_panel.set_values(self._pulsed_input().pulse_params)
        self._load_windows()

    def _load_windows(self):
        flank = float(self._last_flank_ns)
        win = self._current_windows()
        self._window_panel.set_values(win, flank)
        self._window_panel.set_flank(flank, self._flank_detected)
        self._loaded_windows = self._window_panel.values()
        ref0 = win.get('ref_delay_ns')
        self._trace_panel.set_windows(win['detect_delay_ns'] - flank, win['detect_ns'],
                                      None if ref0 is None else ref0 - flank,
                                      win.get('ref_ns') or None)

    def _to_gate_time(self, values):
        flank = float(self._last_flank_ns)
        out = dict(values)
        for key in FLANK_FIELDS:
            if key in out:
                out[key] = int(round(out[key] + flank))
        return out

    @QtCore.Slot(dict)
    def _apply_sequence(self, values):
        if self._scan_running():
            self._sequence_panel.set_status('Stop the scan first; the sequence changes between '
                                            'sweeps.', error=True)
            return
        try:
            self._pulsed_input().set_pulse_params(**values)
        except Exception as err:
            self._sequence_panel.set_status(f'Not applied: {err}', error=True)
            return
        self._sequence_panel.set_values(self._pulsed_input().pulse_params)
        self._sequence_panel.set_status('Applied; used from the next sweep start.')

    @QtCore.Slot(dict)
    def _windows_edited(self, values):
        """ Window fields or regions changed: live re-analysis with software windows, else a
        hardware-gate change between sweeps. """
        gate = self._to_gate_time(values)
        self._trace_panel.set_windows(values['detect_delay_ns'], values['detect_ns'],
                                      values.get('ref_delay_ns'), values.get('ref_ns'))
        if self._live:
            self._window_panel.set_status('applying...')
            self.sigSetAnalysisWindows.emit(gate)
            return
        if self._scan_running():
            self._window_panel.allow_resend()     # Enter again after the scan applies them
            self._window_panel.set_status('Hardware windows: stop the scan first, then press '
                                          'Enter in a field.', error=True)
            return
        try:
            self._pulsed_input().set_pulse_params(**gate)
        except Exception as err:
            self._window_panel.allow_resend()
            self._window_panel.set_status(f'Not applied: {err}', error=True)
            return
        self._load_windows()
        self._window_panel.set_status('Applied (hardware windows); used from the next sweep.')

    @QtCore.Slot(dict)
    def _windows_applied(self, result):
        if self._window_panel is None:
            return
        if 'error' in result:
            self._window_panel.set_status(f"Not applied: {result['error']}", error=True)
            self._load_windows()
            return
        self._loaded_windows = self._window_panel.values()
        frames = result.get('frames', 0)
        how = f're-analysed {frames} sweeps' if frames else 'used for new data'
        self._window_panel.set_status(
            f"Applied live ({how}); readout from {result.get('detect_delay_ns')} ns after the "
            f"laser gate edge.")

    @QtCore.Slot()
    def _reset_trace(self):
        self._pulsed_input().reset_time_trace()
        self._refresh()

    @QtCore.Slot()
    def _refresh(self):
        if self._trace_panel is None:
            return
        self._sequence_panel.apply_button.setEnabled(not self._scan_running())
        if self._tabs is not None and self._tabs.currentIndex() != 1:
            return      # trace only drawn while its page is shown
        try:
            trace = self._pulsed_input().get_time_trace()
        except Exception as err:
            self._trace_panel.info_label.setText(f'time trace unavailable: {err}')
            return
        flanks = detect_flanks(trace['t_ns'], trace['counts'])
        if not flank_is_trustworthy(trace['counts'], flanks):
            flanks = None                           # keep the last good flank
        if flanks is not None:
            flank = float(np.asarray(trace['t_ns'])[flanks[0]])
            if abs(flank - float(self._last_flank_ns)) >= 1.0 or not self._flank_detected:
                unedited = self._window_panel.values() == self._loaded_windows
                old = float(self._last_flank_ns)
                self._last_flank_ns = flank
                self._flank_detected = True
                if unedited:
                    self._load_windows()           # same gate-time windows, new flank frame
                else:
                    self._window_panel.set_flank(flank, True)
                    self._window_panel.set_status(
                        f'Laser flank moved {old:.0f} -> {flank:.0f} ns while the fields were '
                        f'being edited; the next change uses {flank:.0f} ns.')
        snapshot = self._histogram_snapshot()
        self._trace_panel.set_points(None if snapshot is None else snapshot['frequencies'])
        index = self._trace_panel.point_index
        if snapshot is not None and 0 <= index < snapshot['hist'].shape[0]:
            # one frequency point, summed over the finished sweeps; flank from the full sum
            row = np.asarray(snapshot['hist'][index], dtype=np.int64)
            t_ns = np.arange(row.size) * snapshot['binwidth_ns'] - snapshot['pre_ns']
            trace = dict(t_ns=t_ns, counts=row, rows=row[None, :],
                         binwidth_ns=snapshot['binwidth_ns'],
                         label=f"at point {index}, {snapshot['frames']} sweeps")
        self._trace_panel.set_trace(trace, flanks, float(self._last_flank_ns))
        if self._rates_panel is not None:
            self._rates_panel.set_rates(self._point_rates())

    def _histogram_snapshot(self):
        logic = self._odmr_logic()
        if not self._live or not hasattr(logic, 'histogram_snapshot'):
            return None
        try:
            return logic.histogram_snapshot()
        except Exception:
            return None

    def _point_rates(self):
        logic = self._odmr_logic()
        if not self._live or not hasattr(logic, 'point_rates'):
            return None
        try:
            return logic.point_rates()
        except Exception:
            return None
