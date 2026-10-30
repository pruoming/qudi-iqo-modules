# -*- coding: utf-8 -*-
"""
g2 (Hanbury Brown-Twiss) measurement on one spot: live correlation histogram of two detectors, fit of the
antibunching dip, background correction and saving.

Written for the confocal_nv bench, 2026-10-30 (RUN-CONFOCAL-NV-G2-MODULE-STEP2-20261030), from the bench
script agent-pipeline/temp/g2_20261030/g2.py (tested on B3_-1_2). Plan:
agent-pipeline/temp/g2_module_20261030/G2_MODULE_PLAN.md.

Rules
  * Start refuses (nothing sent) while a CW ODMR, pulsed ODMR, pulsed measurement, confocal scan, optimize
    or autofocus runs (the connected logics are read, never commanded, except the optimizer for the
    optional refocus).
  * Laser: through the confocal scanner's laser_on()/laser_off() (laser_gate connector). The laser state
    before the run is recorded; the laser is switched off at the end only if this module switched it on.
    If laser_off() is refused (a scan is running), it is retried until it succeeds. laser_off() sets the
    whole shared Pulse Streamer LOW, so it is never sent while another measurement is busy; the switch-off
    waits until all are idle. A laser the operator lit is asserted again at the start (its flag can be
    stale after an ODMR or pulsed run) and left on. Without a laser gate the operator is responsible for
    the light (logged). A run whose start rate stays below `min_start_rate` is aborted (laser off?).
  * Watchdog on every poll: the run is aborted (data kept and saved, marked "aborted") if another of those
    measurements starts, or if the summed count rate stays below `watchdog_fraction` of the start rate for
    two polls (laser or spot lost: accumulating on would look like background).
  * Optional refocus (off by default; refused while the periodic autofocus is on): when the mean rate over
    `refocus_window` s falls below `refocus_fraction` of the reference, the histogram pauses, one optimize
    runs at the current position and the run continues. The laser stays on through laser_on()'s flag.
    The reference is the highest rate of the first 5 s after a start or a refocus (after a refocus at
    least 90 % of the previous one); after `max_refocus` refocuses the run is aborted. Stopping or
    aborting during a refocus also stops that optimize.
  * Errors: any exception while polling aborts the run (counter stopped, laser restored); the module
    keeps one run object, so a laser g2 switched on is still switched off by a later run or poll.
  * No stage moves. The background is measured with "take background" where the operator has put the
    cursor (off the spot), or entered by hand. rho = (on - off) / on, with on = mean rate of the run.
  * A run stops after its run time or by hand; at the end it is fitted and saved (also when aborted).

Example config for copy-paste:

    g2_logic:
        module.Class: 'g2_logic.G2Logic'
        connect:
            correlation_counter: timetagger_correlation
            laser_gate: confocal_scan_io                    # optional
            optimize_logic: scanning_optimize_logic         # optional (refocus, busy check)
            scanning_logic: scanning_probe_logic            # optional (busy check, position in the file)
            odmr_logic: odmr_logic                          # optional (busy check)
            pulsed_odmr_logic: pulsed_odmr_logic            # optional (busy check)
            pulsed_measurement_logic: pulsed_measurement_logic  # optional (busy check)
            autofocus_logic: autofocus_logic                # optional (busy check, refocus exclusion)
        options:
            poll_interval: 1.0          # s
            watchdog_fraction: 0.3
            refocus_fraction: 0.85
            refocus_window: 10.0        # s
            refocus_timeout: 300.0      # s
            max_run_time: 86400.0       # s

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

import datetime
import re
import time

import numpy as np
from PySide6 import QtCore

from qudi.core.configoption import ConfigOption
from qudi.core.connector import Connector
from qudi.core.module import LogicBase
from qudi.core.statusvariable import StatusVar
from qudi.util.datastorage import TextDataStorage
from qudi.util.g2_model import fit_g2, correct, raw_floor, model
from qudi.util.mutex import RecursiveMutex

ACTIVE_PHASES = ('running', 'refocus')
# while the module's own refocus runs, these two are busy because of it
REFOCUS_MODULES = frozenset(('scanning_optimize_logic', 'scanning_probe_logic'))


class G2Run:
    """ One g2 run as a state machine, without the qudi module machinery (tested on its own).

    counter: CorrelationCounterInterface (configured and stopped). laser: object with laser_is_on,
    laser_on(), laser_off(), or None. busy(exclude) -> {name: state} of other running measurements.
    refocus: object with start(), stop(), busy() and allowed(), or None.

    One instance serves all runs of a module, so the laser bookkeeping (who switched it on, a pending
    switch-off) carries over from one run to the next. The reference rate for the watchdog and the
    refocus is the highest rate of the first `ref_window_s` after a start or a refocus. Any exception
    inside poll() aborts the run (counter stopped, laser restored). """

    def __init__(self, counter, *, laser=None, busy=None, refocus=None, clock=time.monotonic, log=None,
                 run_time_s=600.0, watchdog_fraction=0.3, refocus_enabled=False, refocus_fraction=0.85,
                 refocus_window_s=10.0, refocus_timeout_s=300.0, refocus_start_grace_s=5.0,
                 ref_window_s=5.0, max_refocus=10, min_start_rate_cps=2000.0, position=None):
        self.ref_window_s = float(ref_window_s)
        self.max_refocus = int(max_refocus)
        self.min_start_rate_cps = float(min_start_rate_cps)   # below: laser off or no spot (dark ~600 c/s)
        self.position = position                              # callable -> scanner position (um) or None
        self.norm_rr = 0.0          # sum of rate_A * rate_B * dt over the accumulation (own normalisation)
        self._deadline = None       # wall-clock limit of the current segment
        self.counter = counter
        self.laser = laser
        self.busy = busy or (lambda exclude=frozenset(): dict())
        self.refocus = refocus
        self.clock = clock
        self.log = log or (lambda level, msg: None)
        self.run_time_s = float(run_time_s)
        self.watchdog_fraction = float(watchdog_fraction)
        self.refocus_enabled = bool(refocus_enabled)
        self.refocus_fraction = float(refocus_fraction)
        self.refocus_window_s = float(refocus_window_s)
        self.refocus_timeout_s = float(refocus_timeout_s)
        self.refocus_start_grace_s = float(refocus_start_grace_s)
        self.phase = 'idle'
        self.message = ''
        self.reason = ''
        self.ref_rate = None
        self.last_rates = None
        self.rate_log = list()      # (seconds since the first start, summed rate in c/s, poll length in s)
        self.refocus_log = list()
        self.laser_note = ''
        self._t0 = None
        self._last_poll = None
        self._low_polls = 0
        self._window_start = None
        self._refocus = None
        self._ref_until = None          # the reference rate is still being taken until this time
        self._we_lit = False
        self._restore_pending = False
        self._off_error_logged = False
        self._started_wall = None

    # ------------------------------------------------------------ control
    @property
    def active(self):
        return self.phase in ACTIVE_PHASES

    @property
    def restore_pending(self):
        return self._restore_pending

    def start(self, resume=False):
        """ Start a new run (histogram cleared) or continue the current one. Returns (ok, message). """
        if self.active:
            return False, 'a g2 run is already active'
        others = self.busy(frozenset())
        if others:
            return False, f'refused: {", ".join(sorted(others))} running; nothing started'
        if self.refocus_enabled and (self.refocus is None or not self.refocus.allowed()):
            return False, ('refused: the g2 refocus is enabled but not possible (periodic autofocus on, or no '
                           'optimizer connected); disable one of them')
        try:
            ok, message = self.acquire_laser()
            if not ok:
                return False, message
            if not resume:
                self.counter.clear()
                self.rate_log = list()
                self.refocus_log = list()
                self.norm_rr = 0.0
                self._t0 = self.clock()
            elif self._t0 is None:
                self._t0 = self.clock()
            self.counter.get_count_rates()          # start a fresh rate window
            self.counter.start()
        except Exception as exc:
            self.log('error', f'g2 start failed: {exc}')
            self._safe_counter_stop()
            self.release_laser()
            return False, f'start failed: {exc}'
        now = self.clock()
        self._last_poll = now
        self._window_start = now
        self.ref_rate = None
        self._ref_until = now + self.ref_window_s
        try:
            done = float(self.counter.get_data()['capture_s'])
        except Exception:
            done = 0.0
        # wall-clock limit: twice the remaining run time plus 10 min (refocuses, pauses); then abort
        self._deadline = now + 2 * max(0.0, self.run_time_s - done) + 600.0
        self._low_polls = 0
        self.reason = ''
        self._set('running', 'continued' if resume else 'started')
        return True, self.message

    def stop(self, reason='stopped by the operator'):
        if self.active:
            self._end('stopped', reason)

    def abort(self, reason):
        if self.active:
            self._end('aborted', reason)

    def acquire_laser(self):
        """ Make sure the laser is on, remembering whether g2 switched it on. Returns (ok, message). """
        if self.laser is None:
            self.laser_note = 'no laser gate connected: the laser is the operator\'s responsibility'
            self.log('warning', self.laser_note)
            return True, self.laser_note
        if self._we_lit and self.laser.laser_is_on:
            self._restore_pending = False               # still ours from an earlier g2 action
            self.laser.laser_on()                       # the Pulse Streamer may have been reset meanwhile
            self.laser_note = 'laser kept on by g2 (switched on by an earlier g2 action)'
        elif not self.laser.laser_is_on:
            self.laser.laser_on()
            if not self.laser.laser_is_on:
                return False, 'refused: the laser gate did not switch on (scan running?)'
            self._we_lit = True
            self._restore_pending = False
            self.laser_note = 'laser switched on by g2'
        else:
            # the flag says on, but an ODMR or pulsed run since then may have left the Pulse Streamer LOW:
            # assert the laser again without taking ownership (it stays on at the end)
            self.laser.laser_on()
            self.laser_note = 'laser was already on (asserted again; left on at the end)'
        return True, self.laser_note

    def release_laser(self):
        """ Switch the laser off again if g2 switched it on (retried by poll() if refused). """
        self._restore_pending = self._we_lit
        self._restore_laser()

    def reset_rates(self):
        """ After the histogram was cleared during a run: rates and refocus history start again. """
        self.rate_log = list()
        self.refocus_log = list()
        self.norm_rr = 0.0
        self._t0 = self.clock()

    def poll(self):
        """ One poll: watchdog, refocus, run time. Returns the phase. Also retries a pending laser restore.
        An exception aborts the run. """
        now = self.clock()
        try:
            if self.phase == 'running':
                self._poll_running(now)
            elif self.phase == 'refocus':
                self._poll_refocus(now)
        except Exception as exc:
            self.log('error', f'g2 poll failed: {exc}')
            self.abort(f'error: {exc}')
        if not self.active and self._restore_pending:
            self._restore_laser()
        return self.phase

    # ------------------------------------------------------------ internals
    def _poll_running(self, now):
        dt = now - self._last_poll
        self._last_poll = now
        rates = np.asarray(self.counter.get_count_rates(), float)
        total = float(rates.sum())
        self.last_rates = rates
        if dt > 0:
            self.rate_log.append((now - self._t0, total, dt))
            if rates.size >= 2:
                self.norm_rr += float(rates[0] * rates[1]) * dt
        if self._ref_until is not None:     # reference: highest rate of the first ref_window_s
            self.ref_rate = total if self.ref_rate is None else max(self.ref_rate, total)
            if now >= self._ref_until:
                self._ref_until = None
                if self.ref_rate < self.min_start_rate_cps:
                    self._end('aborted', f'count rate {self.ref_rate / 1e3:.2f} kc/s below '
                                         f'{self.min_start_rate_cps / 1e3:.1f} kc/s: laser off or no spot?')
                    return
        if self._deadline is not None and now > self._deadline:
            self._end('aborted', 'wall-clock limit exceeded (capture time not advancing?)')
            return
        others = self.busy(frozenset())
        if others:
            self._end('aborted', f'{", ".join(sorted(others))} started during the g2 run')
            return
        if self.ref_rate > 0 and total < self.watchdog_fraction * self.ref_rate:
            self._low_polls += 1
            if self._low_polls >= 2:
                self._end('aborted', f'count rate fell to {100 * total / self.ref_rate:.0f} % of the start '
                                     f'({total / 1e3:.1f} kc/s): laser or spot lost')
                return
        else:
            self._low_polls = 0
        if (self.refocus_enabled and self.refocus is not None and self._ref_until is None
                and now - self._window_start >= self.refocus_window_s):
            recent = [(r, d) for t, r, d in self.rate_log if t >= now - self._t0 - self.refocus_window_s]
            mean = sum(r * d for r, d in recent) / max(1e-9, sum(d for _, d in recent))
            if mean < self.refocus_fraction * self.ref_rate:
                if len(self.refocus_log) >= self.max_refocus:
                    self._end('aborted', f'rate still below {100 * self.refocus_fraction:.0f} % after '
                                         f'{self.max_refocus} refocuses')
                    return
                self.counter.stop()
                self._refocus = dict(started=now, at_s=round(now - self._t0, 1), rate_before=mean,
                                     seen_busy=False, position_before_um=self._position())
                self.refocus.start()
                self._set('refocus', f'refocus: rate {mean / 1e3:.1f} kc/s below '
                                     f'{100 * self.refocus_fraction:.0f} % of {self.ref_rate / 1e3:.1f} kc/s')
                return
        capture = float(self.counter.get_data()['capture_s'])
        if capture >= self.run_time_s:
            self._end('finished', f'run time {self.run_time_s:.0f} s reached')

    def _poll_refocus(self, now):
        others = self.busy(REFOCUS_MODULES)
        if others:
            self._end('aborted', f'{", ".join(sorted(others))} started during the g2 refocus')
            return
        busy = self.refocus.busy()
        rf = self._refocus
        if busy:
            rf['seen_busy'] = True
        elapsed = now - rf['started']
        if rf['seen_busy'] and not busy:
            rf['duration_s'] = round(elapsed, 1)
            rf['position_after_um'] = self._position()
            self.refocus_log.append({k: v for k, v in rf.items() if k not in ('started', 'seen_busy')})
            self._refocus = None
            self.counter.get_count_rates()      # drop the rates of the refocus time
            self.counter.start()
            self._last_poll = now
            self._window_start = now
            self.ref_rate = 0.9 * self.ref_rate  # new reference: at least 90 % of the old one
            self._ref_until = now + self.ref_window_s
            self._low_polls = 0
            self._set('running', 'continued after the refocus')
        elif not rf['seen_busy'] and elapsed > self.refocus_start_grace_s:
            self._end('aborted', 'the refocus optimize did not start')
        elif elapsed > self.refocus_timeout_s:
            self.refocus.stop()
            self._end('aborted', f'the refocus did not finish within {self.refocus_timeout_s:.0f} s')

    def _end(self, phase, reason):
        """ End the run: phase first (so an error below cannot keep it alive), then counter, own refocus,
        laser. """
        in_refocus = self._refocus is not None
        self._set(phase, reason)
        self.reason = reason
        self._refocus = None
        self._safe_counter_stop()
        if in_refocus and self.refocus is not None:
            try:
                # our optimize: do not leave it scanning. Stopped even if it does not look busy yet: the
                # queued stop follows the queued start, and stop_optimize on an idle optimizer is a no-op
                self.refocus.stop()
            except Exception as exc:
                self.log('error', f'stopping the refocus optimize failed: {exc}')
        self.release_laser()
        self.log('warning' if phase == 'aborted' else 'info', f'g2 run {phase}: {reason}')

    def _safe_counter_stop(self):
        try:
            self.counter.stop()
        except Exception as exc:
            self.log('error', f'stopping the correlation failed: {exc}')

    def _restore_laser(self):
        """ Switch the laser off if g2 switched it on; stays pending (retried by poll) if refused.

        laser_off() sets the WHOLE shared Pulse Streamer to all-LOW: it is never called while another
        measurement (scan, optimize, ODMR, pulsed ODMR, pulsed) is busy, because that would silently kill
        that measurement's sequence. The switch-off then waits until all of them are idle again. """
        if not self._restore_pending or self.laser is None:
            self._restore_pending = False
            return
        try:
            others = self.busy(frozenset())
        except Exception:
            others = {'unknown': 'busy check failed'}
        if others:
            self.laser_note = (f'laser switch-off waits: {", ".join(sorted(others))} uses the Pulse Streamer')
            return
        try:
            self.laser.laser_off()
            still_on = self.laser.laser_is_on
        except Exception as exc:
            if not self._off_error_logged:
                self.log('error', f'laser_off failed: {exc} (retried every poll, logged once)')
                self._off_error_logged = True
            still_on = True
        if still_on:
            self.laser_note = 'laser_off refused (scan running?); retrying'
            return
        self._off_error_logged = False
        self._restore_pending = False
        self._we_lit = False
        self.laser_note = 'laser switched off again by g2'

    def _position(self):
        try:
            return self.position() if self.position is not None else None
        except Exception:
            return None

    def _set(self, phase, message):
        self.phase = phase
        self.message = message

    # ------------------------------------------------------------ summaries
    def rate_summary(self, edge_s=60.0):
        """ Mean summed rate of the whole run and of its first / last `edge_s` (at most 10 % each). """
        if not self.rate_log:
            return dict()
        t = np.array([x[0] for x in self.rate_log])
        r = np.array([x[1] for x in self.rate_log])
        d = np.array([x[2] for x in self.rate_log])
        span = min(edge_s, 0.1 * float(t[-1] - t[0] + d[0])) if len(t) > 1 else float(d[0])
        first = t <= t[0] + span
        last = t >= t[-1] - span
        mean = lambda m: float(np.sum(r[m] * d[m]) / np.sum(d[m]))
        return dict(on_cps=mean(np.ones(t.size, bool)), on_start_cps=mean(first), on_end_cps=mean(last),
                    seconds=float(np.sum(d)))

    def status(self):
        return dict(phase=self.phase, message=self.message, reason=self.reason, ref_rate=self.ref_rate,
                    last_rates=None if self.last_rates is None else [float(x) for x in self.last_rates],
                    run_time_s=self.run_time_s, refocus_enabled=self.refocus_enabled,
                    refocus_count=len(self.refocus_log), laser_note=self.laser_note,
                    restore_pending=self._restore_pending)


def evaluate(data, rates, background=None, norm_rr=None, refocused=False):
    """ Fit and, with a background rate, rho and the corrected values. data from the counter, rates from
    G2Run.rate_summary(), background dict(cps, err_cps) or None.

    The Time Tagger normalises with the mean rates of the whole accumulation (N_A N_B / T), which is biased
    high when the rate changed during the run (refocuses). After a refocus, and when norm_rr (sum of
    rate_A rate_B dt over the polls, G2Run.norm_rr) is given, the histogram is normalised with it instead. """
    tau_ps = np.asarray(data['tau_ps'], float)
    tau_ns = tau_ps / 1e3
    g = np.asarray(data['g_norm'], float)
    normalisation = 'counter (mean rates of the whole run)'
    if refocused and norm_rr and norm_rr > 0 and tau_ps.size > 1:
        bin_s = float(tau_ps[1] - tau_ps[0]) * 1e-12
        g = np.asarray(data['counts'], float) / (norm_rr * bin_s)
        normalisation = 'per-poll rates (rate changed at refocuses)'
    fit = fit_g2(tau_ns, g, data['counts'])
    res = dict(fit=fit, normalisation=normalisation)
    if not background or not rates or not rates.get('on_cps'):
        return res
    on, off = rates['on_cps'], float(background['cps'])
    rho = (on - off) / on
    drift = abs(rates['on_start_cps'] - rates['on_end_cps']) / 2 / on
    poisson = np.sqrt(on * rates['seconds']) / (on * rates['seconds']) if rates['seconds'] > 0 else 0.0
    rho_err = float(drift + poisson + float(background.get('err_cps', 0.0)) / on)
    res.update(rho=float(rho), rho_err=rho_err, raw_floor_single=raw_floor(rho))
    # differences to g2.py: "on" is the mean rate of the whole run (the histogram's own exposure) instead of
    # 10 s windows before and after it, so its Poisson term is smaller; the background error adds linearly
    if fit and rho > 0:
        res['g2c_0'], res['g2c_0_err'] = correct(fit['g2_0'], fit['g2_0_err'], rho, rho_err)
        res['g2c_dip'], res['g2c_dip_err'] = correct(fit['dip_bins'], fit['dip_bins_err'], rho, rho_err)
        res['g2c_dip_tt'], res['g2c_dip_tt_err'] = correct(fit['dip_bins_tt'], fit['dip_bins_tt_err'], rho,
                                                           rho_err)
    return res


def safe_tag(tag):
    """ A save tag usable in a file name: letters, digits, '-', '_' and '.' only. """
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(tag or '')).strip('_.')


def save_g2(root_dir, data, metadata, result, tag='', timestamp=None):
    """ Histogram as a text file (tau_ns, counts, g_norm) with the run and fit in its header, and a png
    (full window and +-50 ns around the dip, with the fit). Returns the data file path. """
    timestamp = timestamp or datetime.datetime.now()
    tau_ns = np.asarray(data['tau_ps'], float) / 1e3
    counts = np.asarray(data['counts'])
    g = np.asarray(data['g_norm'], float)
    fit = (result or {}).get('fit')
    meta = dict(metadata)
    meta['Fit and correction'] = {k: v for k, v in (result or {}).items() if k != 'fit'}
    if fit:
        meta['Fit and correction']['fit'] = fit
    # one float array for all columns: the coincidences are written as whole numbers with '.0f'
    storage = TextDataStorage(root_dir=root_dir, column_formats=('.4f', '.0f', '.6f'))
    path, _, _ = storage.save_data(np.column_stack([tau_ns, counts, g]), metadata=meta,
                                   nametag=f"{tag + '_' if tag else ''}g2", timestamp=timestamp,
                                   column_headers=['tau (ns)', 'coincidences', 'g2 normalised'],
                                   column_dtypes=[float, int, float])
    from matplotlib.figure import Figure        # no pyplot: safe outside the GUI thread
    fig = Figure(figsize=(10, 4))
    for ax, span in zip(fig.subplots(1, 2), (None, 50.0)):
        sel = np.ones(tau_ns.size, bool) if span is None else np.abs(tau_ns - (fit or {}).get('t0_ns', 0)) <= span
        # explicit styles: qudi's matplotlib style would add markers and change the colour cycle (desk test)
        ax.plot(tau_ns[sel], g[sel], linestyle='none', marker='.', markersize=2, color='#1f77b4', label='data')
        if fit:
            ax.plot(tau_ns[sel], model(tau_ns[sel], fit['level'], 1 - fit['g2_0'], fit['a'], fit['t1_ns'],
                                       fit['t2_ns'], fit['t0_ns'], fit['sigma_ns']),
                    linestyle='-', marker='', linewidth=1.5, color='#d62728', label='fit')
        ax.set_xlabel('tau (ns)')
        ax.set_ylabel('g2 (normalised)')
    if fit:
        fig.suptitle(f"raw g2(0) {fit['g2_0_tt']:.2f} (counter level), {fit['g2_0']:.2f} +- "
                     f"{fit['g2_0_err']:.2f} (fit level); t0 {fit['t0_ns']:.2f} ns, t1 {fit['t1_ns']:.1f} ns")
    fig.axes[0].legend(loc='lower right')
    storage.save_thumbnail(fig, path[:-4] if path.endswith('.dat') else path)
    return path


class G2Logic(LogicBase):
    """ g2 measurement on one spot (see the module docstring). """

    _counter = Connector(name='correlation_counter', interface='CorrelationCounterInterface')
    _laser_gate = Connector(name='laser_gate', interface='ConfocalScanIO', optional=True)
    _optimize_logic = Connector(name='optimize_logic', interface='ScanningOptimizeLogic', optional=True)
    _scanning_logic = Connector(name='scanning_logic', interface='ScanningProbeLogic', optional=True)
    _odmr_logic = Connector(name='odmr_logic', interface='OdmrLogic', optional=True)
    _pulsed_odmr_logic = Connector(name='pulsed_odmr_logic', interface='PulsedOdmrLogic', optional=True)
    _pulsed_logic = Connector(name='pulsed_measurement_logic', interface='PulsedMeasurementLogic',
                              optional=True)
    _autofocus_logic = Connector(name='autofocus_logic', interface='AutofocusPriorityLogic', optional=True)

    _poll_interval = ConfigOption(name='poll_interval', default=1.0, missing='nothing')
    _watchdog_fraction = ConfigOption(name='watchdog_fraction', default=0.3, missing='nothing')
    _refocus_fraction = ConfigOption(name='refocus_fraction', default=0.85, missing='nothing')
    _refocus_window = ConfigOption(name='refocus_window', default=10.0, missing='nothing')
    _refocus_timeout = ConfigOption(name='refocus_timeout', default=300.0, missing='nothing')
    _max_run_time = ConfigOption(name='max_run_time', default=86400.0, missing='nothing')
    _min_start_rate = ConfigOption(name='min_start_rate', default=2000.0, missing='nothing')   # c/s

    _run_time = StatusVar(name='run_time', default=600.0)
    _bin_width_ps = StatusVar(name='bin_width_ps', default=None)
    _n_bins = StatusVar(name='n_bins', default=None)
    _refocus_enabled = StatusVar(name='refocus_enabled', default=False)
    _background = StatusVar(name='background', default=None)      # dict(cps, err_cps, seconds, time)

    sigStatusChanged = QtCore.Signal(dict)
    sigDataUpdated = QtCore.Signal(dict)
    sigResultUpdated = QtCore.Signal(dict)
    sigSaved = QtCore.Signal(str)

    _sigStart = QtCore.Signal(bool)
    _sigStop = QtCore.Signal()
    _sigClear = QtCore.Signal()
    _sigFit = QtCore.Signal()
    _sigSave = QtCore.Signal(str)
    _sigBackground = QtCore.Signal(float)
    _sigSettings = QtCore.Signal(dict)
    _sigStartOptimize = QtCore.Signal()
    _sigStopOptimize = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = RecursiveMutex()
        self._run = None
        self._timer = None
        self._bg_timer = None
        self._bg_state = None
        self._result = dict()
        self._connected = list()
        self._status_cache = dict(phase='idle')

    # ------------------------------------------------------------ lifecycle
    def on_activate(self):
        counter = self._counter()
        settings = counter.settings
        if self._bin_width_ps is None or self._n_bins is None:
            self._bin_width_ps, self._n_bins = settings['bin_width_ps'], settings['n_bins']
        else:
            try:
                counter.configure(settings['channels'], self._bin_width_ps, self._n_bins)
            except Exception as exc:
                self.log.warning(f'stored histogram settings refused ({exc}); using the counter settings')
                self._bin_width_ps, self._n_bins = settings['bin_width_ps'], settings['n_bins']
        self._run_time = float(min(max(1.0, float(self._run_time)), float(self._max_run_time)))
        self._run = self._new_run()
        self._timer = QtCore.QTimer()
        self._timer.setInterval(int(1000 * float(self._poll_interval)))
        self._timer.timeout.connect(self._poll)
        self._bg_timer = QtCore.QTimer()
        self._bg_timer.setSingleShot(True)
        self._bg_timer.timeout.connect(self._background_done)
        queued = QtCore.Qt.ConnectionType.QueuedConnection
        self._sigStart.connect(self._start, queued)
        self._sigStop.connect(self._stop, queued)
        self._sigClear.connect(self._clear, queued)
        self._sigFit.connect(self._fit, queued)
        self._sigSave.connect(self._save, queued)
        self._sigBackground.connect(self._start_background, queued)
        self._sigSettings.connect(self._apply_settings, queued)
        optimizer = self._optimize_logic()
        if optimizer is not None:
            self._link(self._sigStartOptimize, optimizer.start_optimize)
            self._link(self._sigStopOptimize, optimizer.stop_optimize)
        if self._laser_gate() is None:
            self.log.warning('No laser gate connected: switch the laser on yourself for a g2 run.')
        self._emit_status()

    def on_deactivate(self):
        with self._lock:
            if self._run is not None and self._run.active:
                self._run.stop('module deactivated')
                self._finish_run()
            self._timer.stop()
            self._bg_timer.stop()
            self._bg_state = None
            if self._run is not None:
                self._run.release_laser()
                if self._run.restore_pending:
                    self.log.error('g2 deactivated while laser_off is refused (scan running): '
                                   'switch the laser off by hand after the scan')
        for sig in (self._sigStart, self._sigStop, self._sigClear, self._sigFit, self._sigSave,
                    self._sigBackground, self._sigSettings):
            try:
                sig.disconnect()
            except (RuntimeError, TypeError):
                pass
        for signal, slot in self._connected:
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        self._connected = list()

    # ------------------------------------------------------------ public API (any thread)
    def start_run(self):
        self._sigStart.emit(False)

    def continue_run(self):
        self._sigStart.emit(True)

    def stop_run(self):
        self._sigStop.emit()

    def clear_data(self):
        self._sigClear.emit()

    def fit(self):
        self._sigFit.emit()

    def save(self, tag=''):
        self._sigSave.emit(str(tag))

    def take_background(self, seconds=10.0):
        """ Measure the summed rate for `seconds` where the cursor is now (move off the spot first). """
        self._sigBackground.emit(float(seconds))

    def set_settings(self, **settings):
        """ run_time (s), bin_width_ps, n_bins, refocus_enabled, background_cps (None clears). """
        self._sigSettings.emit(dict(settings))

    @property
    def status(self):
        """ The last emitted status (a copy; no lock, so a reader never waits for a fit or a save). """
        return dict(self._status_cache)

    @property
    def result(self):
        return dict(self._result)

    @property
    def is_running(self):
        return self._run is not None and self._run.active

    # ------------------------------------------------------------ slots (logic thread)
    def _start(self, resume):
        with self._lock:
            if self._bg_state is not None:
                self.log.error('g2 start refused: background measurement running')
                return
            self._run.run_time_s = float(self._run_time)
            self._run.refocus_enabled = bool(self._refocus_enabled)
            ok, message = self._run.start(resume=resume)
            if not ok:
                self.log.error(f'g2 start {message}')
                if self._run.restore_pending:
                    self._timer.start()         # retry the refused laser_off of the failed start
            else:
                if not resume:
                    self._result = dict()
                    self.sigResultUpdated.emit(dict())      # the GUI drops the old fit
                    self._started_at = datetime.datetime.now()
                self.log.info(f'g2 run {message} (run time {self._run.run_time_s:.0f} s)')
                self._timer.start()
            self._emit_status()

    def _stop(self):
        with self._lock:
            if self._run.active:
                self._run.stop()
                self._finish_run()
            elif self._bg_state is not None:        # Stop also cancels a background measurement
                self._bg_timer.stop()
                self._bg_state = None
                self._run.release_laser()
                self.log.info('g2 background cancelled')
            self._emit_status()

    def _clear(self):
        with self._lock:
            self._counter().clear()
            if self._run.active:
                self._run.reset_rates()     # rho must describe the same exposure as the histogram
            self._result = dict()
            self.sigResultUpdated.emit(dict())      # the GUI drops the old fit
            self._emit_data()

    def _fit(self):
        with self._lock:
            self._result = self._evaluate(self._counter().get_data())
            self.sigResultUpdated.emit(dict(self._result))

    def _save(self, tag):
        with self._lock:
            data = self._counter().get_data()
            if not np.any(data['counts']):
                self.log.error('g2: nothing to save (empty histogram)')
                return
            # always evaluated afresh: the background or the data may have changed since the last fit
            self._result = self._evaluate(data)
            self.sigResultUpdated.emit(dict(self._result))
            try:
                path = save_g2(self.module_default_data_dir, data, self._metadata(data), self._result,
                               safe_tag(tag))
            except Exception as exc:
                self.log.exception(f'g2 save failed: {exc}')
                return
            self.log.info(f'g2 saved: {path}')
            self.sigSaved.emit(path)

    def _apply_settings(self, settings):
        with self._lock:
            if self._run.active and any(k in settings for k in ('bin_width_ps', 'n_bins')):
                self.log.error('g2: histogram settings cannot change during a run')
                return
            if 'run_time' in settings:
                self._run_time = float(min(max(1.0, float(settings['run_time'])), float(self._max_run_time)))
                self._run.run_time_s = self._run_time
            if 'refocus_enabled' in settings:
                self._refocus_enabled = bool(settings['refocus_enabled'])
                if self._run.active:
                    if self._refocus_enabled and not self._refocus_allowed():
                        self._refocus_enabled = False
                        self.log.error('g2 refocus not enabled: periodic autofocus on or no optimizer')
                    self._run.refocus_enabled = self._refocus_enabled
            if 'bin_width_ps' in settings or 'n_bins' in settings:
                counter = self._counter()
                applied = counter.configure(counter.settings['channels'],
                                            int(settings.get('bin_width_ps', self._bin_width_ps)),
                                            int(settings.get('n_bins', self._n_bins)))
                self._bin_width_ps, self._n_bins = applied['bin_width_ps'], applied['n_bins']
                self._result = dict()
                self.sigResultUpdated.emit(dict())      # the GUI drops the old fit
            if 'background_cps' in settings:
                cps = settings['background_cps']
                self._background = None if cps is None else dict(
                    cps=float(cps), err_cps=float(settings.get('background_err_cps', 0.0)), seconds=None,
                    time=datetime.datetime.now().isoformat(timespec='seconds'), source='entered')
                self._refresh_result()
            self._emit_status()
            self._emit_data()

    def _poll(self):
        with self._lock:
            was_active = self._run.active
            self._run.poll()            # never raises: an error aborts the run inside
            if self._bg_state is not None:
                others = self._busy(frozenset())
                if others:
                    self._bg_state.setdefault('interrupted', set()).update(others)
            if was_active and not self._run.active:
                self._finish_run()
            if not self._run.active and not self._run.restore_pending and self._bg_state is None:
                self._timer.stop()
            try:
                self._emit_data()
            except Exception as exc:
                self.log.error(f'g2 data update failed: {exc}')
            self._emit_status()

    def _refresh_result(self):
        """ Re-evaluate a finished run's result after the background changed. """
        if self._run.active or not self._result:
            return
        data = self._counter().get_data()
        if np.any(data['counts']):
            self._result = self._evaluate(data)
            self.sigResultUpdated.emit(dict(self._result))

    def _finish_run(self):
        """ After finished / stopped / aborted: fit and save (data kept either way). """
        data = self._counter().get_data()
        if not np.any(data['counts']):
            return
        try:
            self._result = self._evaluate(data)
            self.sigResultUpdated.emit(dict(self._result))
        except Exception as exc:
            self.log.error(f'g2 fit failed: {exc}')
        try:
            path = save_g2(self.module_default_data_dir, data, self._metadata(data), self._result)
            self.log.info(f'g2 saved: {path}')
            self.sigSaved.emit(path)
        except Exception as exc:
            self.log.exception(f'g2 save failed: {exc}')

    # ------------------------------------------------------------ background
    def _start_background(self, seconds):
        with self._lock:
            if self._run.active or self._bg_state is not None:
                self.log.error('g2 background refused: a g2 run or background measurement is active')
                return
            others = self._busy(frozenset())
            if others:
                self.log.error(f'g2 background refused: {", ".join(sorted(others))} running')
                return
            try:
                ok, message = self._run.acquire_laser()     # same bookkeeping as a run
                if not ok:
                    self.log.error(f'g2 background {message}')
                    return
                self._counter().get_count_rates()
            except Exception as exc:
                self.log.error(f'g2 background failed: {exc}')
                self._run.release_laser()
                self._timer.start()             # retries a refused laser_off
                return
            self._bg_state = dict(seconds=float(seconds), started=time.monotonic())
            self._bg_timer.start(int(1000 * float(seconds)))
            self._timer.start()                 # polls during the background: busy check
            self._emit_status()

    def _background_done(self):
        with self._lock:
            if self._bg_state is None:
                return
            state, self._bg_state = self._bg_state, None
            try:
                others = set(self._busy(frozenset())) | state.get('interrupted', set())
                rates = np.asarray(self._counter().get_count_rates(), float)
                seconds = time.monotonic() - state['started']
                if others:
                    self.log.error(f'g2 background discarded: {", ".join(sorted(others))} ran during it')
                else:
                    cps = float(rates.sum())
                    self._background = dict(
                        cps=cps, err_cps=float(np.sqrt(max(cps, 1.0) * seconds) / seconds),
                        seconds=round(seconds, 1), per_channel=[float(r) for r in rates],
                        time=datetime.datetime.now().isoformat(timespec='seconds'), source='measured',
                        position_um=self._position_um())
                    self.log.info(f'g2 background: {cps / 1e3:.2f} kc/s over {seconds:.1f} s')
                    self._refresh_result()
            except Exception as exc:
                self.log.error(f'g2 background failed: {exc}')
            self._run.release_laser()           # refused laser_off: retried by the poll timer
            self._emit_status()

    # ------------------------------------------------------------ helpers
    def _new_run(self):
        return G2Run(self._counter(), laser=self._laser_gate(), busy=self._busy, refocus=_Refocus(self),
                     log=self._log_from_run, run_time_s=float(self._run_time),
                     watchdog_fraction=float(self._watchdog_fraction), refocus_enabled=bool(self._refocus_enabled),
                     refocus_fraction=float(self._refocus_fraction), refocus_window_s=float(self._refocus_window),
                     refocus_timeout_s=float(self._refocus_timeout),
                     min_start_rate_cps=float(self._min_start_rate), position=self._position_um)

    def _evaluate(self, data):
        run = self._run
        return evaluate(data, run.rate_summary(), self._background, norm_rr=run.norm_rr,
                        refocused=bool(run.refocus_log))

    def _log_from_run(self, level, message):
        getattr(self.log, level)(message)

    def _busy(self, exclude=frozenset()):
        """ Other measurements that are running: {name: state}. Read-only. """
        out = dict()
        for name, connector in (('odmr_logic', self._odmr_logic), ('pulsed_odmr_logic', self._pulsed_odmr_logic),
                                ('pulsed_measurement_logic', self._pulsed_logic),
                                ('scanning_probe_logic', self._scanning_logic),
                                ('scanning_optimize_logic', self._optimize_logic)):
            if name in exclude:
                continue
            module = connector()
            if module is not None:
                try:
                    state = str(module.module_state())
                except Exception:
                    continue
                if state == 'locked':
                    out[name] = state
        autofocus = self._autofocus_logic()
        if autofocus is not None and 'autofocus_logic' not in exclude:
            try:
                if autofocus.is_running:
                    out['autofocus_logic'] = 'running'
            except Exception:
                pass
        return out

    def _refocus_allowed(self):
        if self._optimize_logic() is None:
            return False
        autofocus = self._autofocus_logic()
        try:
            return autofocus is None or not autofocus.periodic_enabled
        except Exception:
            return False

    def _position_um(self):
        scanning = self._scanning_logic()
        if scanning is None:
            return None
        try:
            target = dict(scanning.scanner_target)      # a copy: never edit scanner_target in place
            return {k: round(float(v) * 1e6, 4) for k, v in target.items()}
        except Exception:
            return None

    def _metadata(self, data):
        run = self._run
        return {'Module': 'g2_logic', 'Status': run.phase, 'Reason': run.reason,
                'Started': getattr(self, '_started_at', datetime.datetime.now()).isoformat(timespec='seconds'),
                'Capture time (s)': round(float(data['capture_s']), 3),
                'Run time setting (s)': run.run_time_s, 'Counter settings': self._counter().settings,
                'Rates (c/s)': run.rate_summary(), 'Start reference rate (c/s)': run.ref_rate,
                'Background': self._background, 'Refocus enabled': run.refocus_enabled,
                'Refocus': run.refocus_log, 'Laser': run.laser_note,
                'Scanner position at save (um)': self._position_um()}

    def _status(self):
        st = self._run.status() if self._run is not None else dict(phase='idle')
        st.update(background=self._background, bin_width_ps=self._bin_width_ps, n_bins=self._n_bins,
                  run_time_s=float(self._run_time), refocus_enabled=bool(self._refocus_enabled),
                  background_running=self._bg_state is not None,
                  laser_gate=self._laser_gate() is not None, refocus_possible=self._refocus_allowed())
        return st

    def _emit_status(self):
        self._status_cache = self._status()
        self.sigStatusChanged.emit(dict(self._status_cache))

    def _emit_data(self):
        data = self._counter().get_data()
        self.sigDataUpdated.emit(dict(tau_ns=np.asarray(data['tau_ps'], float) / 1e3,
                                      g_norm=np.asarray(data['g_norm'], float),
                                      counts=np.asarray(data['counts']), capture_s=float(data['capture_s']),
                                      rates=None if self._run.last_rates is None else list(self._run.last_rates)))

    def _link(self, signal, slot):
        signal.connect(slot, QtCore.Qt.ConnectionType.QueuedConnection)
        self._connected.append((signal, slot))


class _Refocus:
    """ The optimizer as seen by G2Run (start/stop through queued signals, busy read-only). """

    def __init__(self, logic):
        self._logic = logic

    def start(self):
        self._logic._sigStartOptimize.emit()

    def stop(self):
        self._logic._sigStopOptimize.emit()

    def busy(self):
        optimizer = self._logic._optimize_logic()
        return optimizer is not None and str(optimizer.module_state()) == 'locked'

    def allowed(self):
        return self._logic._refocus_allowed()
