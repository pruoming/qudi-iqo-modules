# -*- coding: utf-8 -*-
"""
Pulsed ODMR logic with software analysis windows (2026-10-24).

Runs exactly like OdmrLogic (frequency sweeps, raw per-sweep data, fits, saving), but when the
data scanner records per-point photon-arrival histograms (PulsedOdmrScanInput with
software_windows) it also SUMS those histograms over all sweeps and computes the spectrum from
them with the CURRENT readout/reference windows. set_analysis_windows() changes the windows at
any time, also during a scan; the whole accumulated data set is re-analysed at once, like the
analysis windows of the pulsed measurement GUI.

Notes:
* The averaged signal (and the saved signal file) always follows the current windows. The
  per-sweep raw image keeps, for each sweep, the values computed with the windows in force when
  that sweep finished.
* scans_to_average is not applied to the histogram path: the spectrum uses all sweeps.
* save_odmr_data additionally saves the summed per-point histograms, so any window can be
  applied offline later.

Example config:

    pulsed_odmr_logic:
        module.Class: 'pulsed_odmr_logic.PulsedOdmrLogic'
        connect:
            microwave: mw_source_smiq
            data_scanner: pulsed_odmr_scan_input
        options:
            default_scan_mode: 'JUMP_LIST'
            reference_channel: 'fluorescence_ref'
"""

import datetime

import numpy as np
from PySide6 import QtCore

from qudi.logic.odmr_logic import OdmrLogic
from qudi.util.datastorage import TextDataStorage


def window_bins(start_ns, width_ns, binwidth_ns, pre_ns):
    """ Histogram bin slice [a, b) of a window given in ns after the readout laser gate edge;
    bin i starts at i * binwidth - pre. Same rule as PulsedOdmrScanInput._window_bins. """
    a = int(round((start_ns + pre_ns) / binwidth_ns))
    return a, max(a + 1, int(round((start_ns + width_ns + pre_ns) / binwidth_ns)))


def window_rates(hist, shots, windows):
    """ (readout, reference) rates in c/s per point row of summed histograms. """
    bw, pre = windows['binwidth_ns'], windows['pre_ns']
    a0, a1 = window_bins(windows['detect_delay_ns'], windows['detect_ns'], bw, pre)
    r0, r1 = window_bins(windows['ref_delay_ns'], windows['ref_ns'], bw, pre)
    hist = np.asarray(hist, dtype=np.float64)
    # a window reaching past the recorded bins (readout lengthened since) counts what exists
    n = hist.shape[1]
    a1, r1 = min(a1, n), min(r1, n)
    a0, r0 = min(a0, a1 - 1), min(r0, r1 - 1)
    shots = max(float(shots), 1.0)
    readout = hist[:, a0:a1].sum(axis=1) / (shots * (a1 - a0) * bw * 1e-9)
    reference = hist[:, r0:r1].sum(axis=1) / (shots * (r1 - r0) * bw * 1e-9)
    return readout, reference


class PulsedOdmrLogic(OdmrLogic):
    """ OdmrLogic that re-analyses per-point histograms with live software windows. """

    sigAnalysisWindowsUpdated = QtCore.Signal(dict)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hist_sum = None        # (points, bins) summed over sweeps, all ranges joined
        self._hist_shots = 0         # shots per point summed over sweeps
        self._hist_frames = 0
        self._hist_meta = None       # (binwidth_ns, pre_ns, readout_ns, bins) of _hist_sum
        # read by the GUI without the logic lock (held for a whole sweep line): one tuple,
        # replaced (never modified) after each absorbed sweep
        self._hist_snapshot = None   # (hist, shots, frames, binwidth_ns, pre_ns, frequencies)

    # -- data life cycle -----------------------------------------------------------------

    def _discard_pending_frame(self):
        """ A frame left over from an aborted sweep must not enter a new or resumed scan. """
        try:
            self._data_scanner().pop_frame_histogram()
        except Exception:
            pass

    def _initialize_odmr_data(self):
        super()._initialize_odmr_data()
        self._hist_sum, self._hist_shots, self._hist_frames, self._hist_meta = None, 0, 0, None
        self._hist_snapshot = None
        self._discard_pending_frame()

    def _arm_scan_hardware(self):
        self._discard_pending_frame()
        super()._arm_scan_hardware()

    @property
    def histogram_info(self):
        """ Sweeps and shots per point in the summed histograms (0 when not in use). """
        return dict(frames=self._hist_frames, shots_per_point=self._hist_shots,
                    active=self._hist_sum is not None)

    def histogram_snapshot(self):
        """ Summed per-point histograms of all absorbed sweeps, for display; lock-free.
        dict(hist (points, bins), shots, frames, binwidth_ns, pre_ns, frequencies) or None. """
        snap = self._hist_snapshot
        if snap is None:
            return None
        hist, shots, frames, bw, pre, freqs = snap
        return dict(hist=hist, shots=shots, frames=frames, binwidth_ns=bw, pre_ns=pre,
                    frequencies=freqs)

    def point_rates(self, windows=None):
        """ (frequencies, readout c/s, reference c/s) per point from the summed histograms with
        the given windows (ns after the laser gate edge; default: the scanner's current ones),
        or None before the first absorbed sweep. Lock-free, for the GUI. """
        snap = self.histogram_snapshot()
        if snap is None:
            return None
        win = dict(self._data_scanner().analysis_windows if windows is None else windows)
        win['binwidth_ns'], win['pre_ns'] = snap['binwidth_ns'], snap['pre_ns']
        readout, reference = window_rates(snap['hist'], snap['shots'], win)
        return snap['frequencies'], readout, reference

    def _absorb_frame(self):
        """ Add the data scanner's last finished frame to the histogram sums. """
        try:
            frame = self._data_scanner().pop_frame_histogram()
        except AttributeError:
            return
        except Exception:
            self.log.exception('Could not read the per-point histograms of the last sweep:')
            return
        if frame is None:
            return
        hist = np.asarray(frame['hist'], dtype=np.int64)
        ovs = max(1, int(self._oversampling_factor))
        n_points = sum(int(r[-1]) for r in self._scan_frequency_ranges)
        if hist.ndim != 2 or hist.shape[0] != n_points * ovs:
            self.log.warning(f'Per-point histograms have {hist.shape[0]} rows for {n_points} '
                             f'frequency points x {ovs} oversampling; sweep not added to the '
                             f'histogram sums.')
            return
        if ovs > 1:
            hist = hist.reshape(n_points, ovs, hist.shape[1]).sum(axis=1)
        # pi/gap/wait in the key too: sweeps with another pulse must not be summed together
        meta = (frame['binwidth_ns'], frame['pre_ns'], frame['readout_ns'], hist.shape[1],
                tuple(frame.get('pulse', ())))
        if self._hist_sum is None or meta != self._hist_meta:
            if self._hist_sum is not None:
                self.log.warning('Histogram geometry changed during the scan; the histogram '
                                 'sums restart from this sweep.')
            self._hist_sum = np.zeros_like(hist)
            self._hist_meta, self._hist_shots, self._hist_frames = meta, 0, 0
        self._hist_sum = self._hist_sum + hist     # new array: a snapshot is never modified
        self._hist_shots += int(frame['shots']) * ovs
        self._hist_frames += 1
        self._hist_snapshot = (self._hist_sum, self._hist_shots, self._hist_frames, meta[0],
                               meta[1], np.concatenate(self._frequency_data))

    def _calculate_signal_data(self):
        self._absorb_frame()
        if self._hist_sum is None:
            super()._calculate_signal_data()
            return
        self._signal_from_histograms()
        self._normalize_signal_data()

    def _signal_from_histograms(self):
        """ Spectrum from the summed histograms with the data scanner's current windows, on
        the time axis the sums were recorded with. Channel order as the data scanner's:
        (readout, reference). """
        windows = dict(self._data_scanner().analysis_windows)
        windows['binwidth_ns'], windows['pre_ns'] = self._hist_meta[0], self._hist_meta[1]
        readout, reference = window_rates(self._hist_sum, self._hist_shots, windows)
        readout_ch, reference_ch = self._data_scanner().constraints.channel_names[:2]
        start = 0
        for range_index, range_params in enumerate(self._scan_frequency_ranges):
            stop = start + int(range_params[-1])
            self._signal_data[readout_ch][range_index] = readout[start:stop].copy()
            self._signal_data[reference_ch][range_index] = reference[start:stop].copy()
            start = stop

    # -- live windows --------------------------------------------------------------------

    @QtCore.Slot(dict)
    def set_analysis_windows(self, windows):
        """ Change the readout/reference windows (ns after the readout laser gate edge) and
        re-analyse all accumulated sweeps. Runs in the logic thread between sweeps. """
        with self._threadlock:
            try:
                applied = dict(self._data_scanner().set_analysis_windows(**windows))
            except Exception as err:
                self.log.error(f'Analysis windows not applied: {err}')
                self.sigAnalysisWindowsUpdated.emit({'error': str(err)})
                return
            if self._hist_sum is not None:
                self._signal_from_histograms()
                self._normalize_signal_data()
                self.sigScanDataUpdated.emit()
            applied.update(self.histogram_info)
            self.sigAnalysisWindowsUpdated.emit(applied)

    # -- saving --------------------------------------------------------------------------

    @QtCore.Slot(str)
    def save_odmr_data(self, tag=None):
        """ Save as OdmrLogic, plus the summed per-point histograms (rows = frequency points,
        columns = time bins) so the analysis can be redone with other windows. """
        super().save_odmr_data(tag)
        with self._threadlock:
            if self._hist_sum is None:
                return
            windows = dict(self._data_scanner().analysis_windows)
            bw, pre, readout_ns, n_bins = self._hist_meta[:4]
            metadata = {'Frequencies (Hz)': list(np.concatenate(self._frequency_data)),
                        'Bin width (ns)': bw, 'Time of first bin (ns after laser gate edge)': -pre,
                        'Readout laser (ns)': readout_ns, 'Sweeps': self._hist_frames,
                        'Shots per point (all sweeps)': self._hist_shots,
                        'Analysis windows (ns after laser gate edge)': {
                            k: windows[k] for k in ('detect_delay_ns', 'detect_ns',
                                                    'ref_delay_ns', 'ref_ns')}}
            nametag = f"{tag + '_' if tag else ''}ODMR_point_histograms"
            TextDataStorage(root_dir=self.module_default_data_dir, column_formats='d').save_data(
                self._hist_sum, metadata=metadata, nametag=nametag,
                timestamp=datetime.datetime.now(),
                column_headers=[f'{-pre + i * bw} ns' for i in range(n_bins)],
                column_dtypes=[int] * n_bins)
