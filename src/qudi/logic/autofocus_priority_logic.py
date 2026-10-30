# -*- coding: utf-8 -*-
"""
Autofocus with priority over the other measurements (pi3diamond-style job priority).

Written for the confocal_nv bench, 2026-10-10 (RUN-CONFOCAL-NV-AUTOFOCUS-20261010), on the operator's
request: during long measurements refocus periodically; while the autofocus runs every other
measurement that shares the instruments pauses and resumes afterwards.

Priority rules
  1. Autofocus (highest). Triggered by a periodic timer (disabled at activation; the operator enables
     it) or by an explicit "autofocus now" request. No count-rate trigger or gating (operator
     decision: some samples degrade slowly, so a count change is not a focus criterion).
  2. Pulsed measurement and ODMR scan: paused for the autofocus and resumed afterwards
       - pulsed: PulsedMeasurementLogic.pause_pulsed_measurement / continue_pulsed_measurement
         (fast counter paused, pulser off, external microwave off; on continue the pulser re-streams
         its stored sequence and the data keep accumulating). A measurement the operator had already
         paused is left paused; one the operator stops during the autofocus is not resumed.
       - ODMR: OdmrLogic.stop_odmr_scan / continue_odmr_scan (stops after the running line; the
         accumulated data are kept; continue re-arms the microwave scan - needs the odmr_logic fix
         of RUN-CONFOCAL-NV-ODMR-RESUME-20261010).
       - ODMR CW output (if the microwave connector is given): switched off through
         OdmrLogic.toggle_cw_output and switched on again afterwards. A microwave output the
         autofocus cannot control (front panel, pulsed without "external microwave") is only noted.
  3. Confocal image scans (any ScanningProbeLogic scan, zoom-ins included) are never interrupted:
     the autofocus waits until the scan has finished (operator decision). There is no timeout;
     "cancel" ends a waiting autofocus.
  The time series (photon counter stream) is not touched.
  The POI Manager's own periodic refocus ("Track") must be off: two refocus loops would fight over the
  optimizer. An autofocus requested while it is on is recorded as failed.

The refocus itself is ScanningOptimizeLogic.start_optimize at the CURRENT scanner position (the
optimizer clamps its fit to its own scan window; its default range is 1 % of each axis travel, 1 um on
the confocal_nv bench, so there one autofocus shifts at most ~0.87 um and max_shift only matters for a
wider optimizer range). The result
is "failed" if the optimizer fit fails, returns nothing, times out, a measurement is started during
the refocus, or any optimizer step's Gaussian fit has R^2 below ``min_fit_rsquared`` (no clear spot in
the window: a fit on noise - laser off, NV bleached or lost - would otherwise move the stage to a random
point; R^2 is scale-free, so it does not depend on the count level of a slowly fading sample); "rejected" if the total shift exceeds ``max_shift``; "cancelled" on operator cancel. For
every result except "accepted" the scanner is moved back to the start position and the POI list is not
changed. An accepted result moves the POI list (ROI) of the POI manager (if connected), but only if the
scanner was at the active POI (within ``poi_tolerance``) when the autofocus started. Every autofocus
(every path, also failures) is appended to a CSV log in the module's data directory and kept in a
StatusVar history.

Example config for copy-paste:

    autofocus_logic:
        module.Class: 'autofocus_priority_logic.AutofocusPriorityLogic'
        options:
            period: 600            # s, default timer period (changeable in the GUI, remembered)
            max_shift: 2.0e-6      # m, total 3D shift above which a result is rejected; null disables
            poi_tolerance: 1.0e-6  # m, max distance scanner - active POI for a POI update
            min_fit_rsquared: 0.5  # optimizer fits below this R^2 -> failed, back to start; 0 disables
            settle_time: 0.5       # s, wait after the other measurements have paused
            optimize_timeout: 300  # s, the optimizer is stopped after this time
            phase_timeout: 60      # s, limit for pausing / resuming the other measurements
        connect:
            optimize_logic: 'scanning_optimize_logic'
            scanning_logic: 'scanning_probe_logic'
            poi_manager_logic: 'poi_manager_logic'                # optional
            pulsed_measurement_logic: 'pulsed_measurement_logic'  # optional
            odmr_logic: 'odmr_logic'                              # optional
            microwave: 'microwave_source'                         # optional, read-only use (CW state)
"""

import csv
import datetime
import math
import os
import time

import numpy as np
from PySide6 import QtCore

from qudi.core.configoption import ConfigOption
from qudi.core.connector import Connector
from qudi.core.module import LogicBase
from qudi.core.statusvariable import StatusVar
from qudi.util.mutex import RecursiveMutex


class AutofocusPriorityLogic(LogicBase):
    """ Periodic / on-demand autofocus that pauses lower-priority measurements while it runs. """

    # connectors
    _optimize_logic = Connector(name='optimize_logic', interface='ScanningOptimizeLogic')
    _scanning_logic = Connector(name='scanning_logic', interface='ScanningProbeLogic')
    _poi_manager_logic = Connector(name='poi_manager_logic', interface='PoiManagerLogic', optional=True)
    _pulsed_logic = Connector(name='pulsed_measurement_logic', interface='PulsedMeasurementLogic',
                              optional=True)
    _odmr_logic = Connector(name='odmr_logic', interface='OdmrLogic', optional=True)
    _microwave = Connector(name='microwave', interface='MicrowaveInterface', optional=True)

    # config options
    _default_period = ConfigOption(name='period', default=600.0, missing='nothing')
    _max_shift = ConfigOption(name='max_shift', default=2.0e-6, missing='nothing')
    _poi_tolerance = ConfigOption(name='poi_tolerance', default=1.0e-6, missing='nothing')
    _min_fit_rsquared = ConfigOption(name='min_fit_rsquared', default=0.5, missing='nothing')
    _settle_time = ConfigOption(name='settle_time', default=0.5, missing='nothing')
    _optimize_timeout = ConfigOption(name='optimize_timeout', default=300.0, missing='nothing')
    _phase_timeout = ConfigOption(name='phase_timeout', default=60.0, missing='nothing')

    # status variables
    _period = StatusVar(name='period', default=None)
    _history = StatusVar(name='history', default=list())

    # public signals
    sigStatusChanged = QtCore.Signal(dict)
    sigHistoryUpdated = QtCore.Signal(list)

    # internal signals (queued into this module's thread)
    _sigTrigger = QtCore.Signal(str)
    _sigCancel = QtCore.Signal()
    _sigSetEnabled = QtCore.Signal(bool)
    _sigSetPeriod = QtCore.Signal(float)
    # commands to the other modules (queued into their threads)
    _sigPausePulsed = QtCore.Signal()
    _sigContinuePulsed = QtCore.Signal()
    _sigStopOdmr = QtCore.Signal()
    _sigContinueOdmr = QtCore.Signal()
    _sigOdmrCw = QtCore.Signal(bool)
    _sigStartOptimize = QtCore.Signal()
    _sigStopOptimize = QtCore.Signal()

    _MIN_PERIOD = 10.0
    _MAX_HISTORY = 500
    _POLL_MS = 250
    _REFUSED_AFTER = 3.0   # s without the optimizer locking: its start was refused
    _MAX_REFUSED = 5       # refused optimizer starts before the autofocus gives up
    _DEACTIVATE_WAIT = 3.0  # s, wait for pending pauses at deactivation before resuming anyway

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = RecursiveMutex()
        self._enabled = False
        self._run = None
        self._next_due = None
        self._phase = 'idle'
        self._message = ''
        self._pulsed_paused = False
        self._period_timer = None
        self._poll_timer = None
        self._connected = list()

    # ------------------------------------------------------------------ lifecycle
    def on_activate(self):
        if self._period is None or not self._period > 0:
            self._period = float(self._default_period)
        if self._period < self._MIN_PERIOD:
            self.log.warning(f'Autofocus period {self._period} s is below {self._MIN_PERIOD:.0f} s; '
                             f'using {self._MIN_PERIOD:.0f} s.')
            self._period = self._MIN_PERIOD
        if not isinstance(self._history, list):
            self._history = list()
        self._enabled = False
        self._run = None
        self._next_due = None
        self._phase = 'idle'
        self._message = ''

        self._period_timer = QtCore.QTimer()
        self._period_timer.setSingleShot(True)
        self._period_timer.timeout.connect(self._period_elapsed)
        self._poll_timer = QtCore.QTimer()
        self._poll_timer.setInterval(self._POLL_MS)
        self._poll_timer.timeout.connect(self._poll)

        queued = QtCore.Qt.ConnectionType.QueuedConnection
        self._sigTrigger.connect(self._trigger, queued)
        self._sigCancel.connect(self._cancel, queued)
        self._sigSetEnabled.connect(self._set_enabled, queued)
        self._sigSetPeriod.connect(self._set_period, queued)

        self._connected = list()
        optimizer = self._optimize_logic()
        self._link(optimizer.sigOptimizeStateChanged, self._optimizer_state_changed)
        self._link(self._sigStartOptimize, optimizer.start_optimize)
        self._link(self._sigStopOptimize, optimizer.stop_optimize)
        pulsed = self._pulsed_logic()
        if pulsed is not None:
            self._pulsed_paused = self._read_private(pulsed, '_PulsedMeasurementLogic__is_paused')
            self._link(pulsed.sigMeasurementStatusUpdated, self._pulsed_status_changed)
            self._link(self._sigPausePulsed, pulsed.pause_pulsed_measurement)
            self._link(self._sigContinuePulsed, pulsed.continue_pulsed_measurement)
        odmr = self._odmr_logic()
        if odmr is not None:
            self._link(self._sigStopOdmr, odmr.stop_odmr_scan)
            self._link(self._sigContinueOdmr, odmr.continue_odmr_scan)
            self._link(self._sigOdmrCw, odmr.toggle_cw_output)
        self._emit_status()
        self.sigHistoryUpdated.emit(list(self._history))

    def on_deactivate(self):
        with self._lock:
            run = self._run
            if run is not None:
                # Leave the other measurements as found: stop the optimizer and resume what we paused.
                self.log.warning('Autofocus logic deactivated during an autofocus; resuming paused measurements.')
                if run['phase'] in ('focusing', 'stopping_optimizer'):
                    self._sigStopOptimize.emit()
                try:
                    if not run['resume_sent']:
                        self._send_resume(run)
                    # no poll after deactivation: wait (bounded) for pending pauses to take effect
                    deadline = time.time() + self._DEACTIVATE_WAIT
                    while run['deferred'] and time.time() < deadline:
                        time.sleep(0.05)
                        self._send_deferred(run)
                    if run['deferred']:
                        self.log.error('Autofocus: pause of ' + ', '.join(run['deferred'])
                                       + ' not confirmed at deactivation; resume sent anyway - check it.')
                        self._send_deferred(run, force=True)
                except Exception:
                    self.log.exception('Autofocus: could not resume the paused measurements.')
                run['result'] = run['result'] or 'aborted'
                self._add_note(run, 'autofocus module deactivated during the autofocus')
                try:
                    self._complete_positions(run)
                    self._record(run)
                except Exception:
                    self.log.exception('Autofocus: could not record the aborted autofocus.')
        for timer in (self._period_timer, self._poll_timer):
            if timer is not None:
                timer.stop()
                timer.timeout.disconnect()
        self._sigTrigger.disconnect()
        self._sigCancel.disconnect()
        self._sigSetEnabled.disconnect()
        self._sigSetPeriod.disconnect()
        for signal, slot in self._connected:
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        self._connected = list()
        self._run = None
        self._enabled = False

    # ------------------------------------------------------------------ public API (any thread)
    def autofocus_now(self):
        """ Request an autofocus as soon as possible (waits for a running image scan). """
        self._sigTrigger.emit('manual')

    def cancel(self):
        """ Cancel a waiting or running autofocus (returns to the start position, resumes the others). """
        self._sigCancel.emit()

    def set_periodic_enabled(self, enabled):
        """ Enable or disable the periodic autofocus timer. """
        self._sigSetEnabled.emit(bool(enabled))

    def set_period(self, period):
        """ Set the timer period in seconds (>= 10 s). """
        self._sigSetPeriod.emit(float(period))

    @property
    def period(self):
        return float(self._period)

    @property
    def periodic_enabled(self):
        return bool(self._enabled)

    @property
    def is_running(self):
        return self._run is not None

    @property
    def history(self):
        return list(self._history)

    def status(self):
        with self._lock:
            return {'enabled': self._enabled,
                    'period': float(self._period),
                    'running': self._run is not None,
                    'phase': self._phase,
                    'next_due': self._next_due,
                    'message': self._message}

    # ------------------------------------------------------------------ timer handling
    def _set_enabled(self, enabled):
        with self._lock:
            self._enabled = bool(enabled)
            if self._enabled:
                if self._run is None:
                    self._restart_period_timer()
                self.log.info(f'Periodic autofocus enabled, period {self._period:.0f} s.')
            else:
                self._period_timer.stop()
                self._next_due = None
                self.log.info('Periodic autofocus disabled.')
                run = self._run
                if run is not None and run['reason'] == 'timer' and run['phase'] == 'waiting_for_scan':
                    run['result'] = 'cancelled'
                    self._add_note(run, 'periodic autofocus disabled while waiting')
                    self._finish()
        self._emit_status()

    def _set_period(self, period):
        with self._lock:
            if not period >= self._MIN_PERIOD:
                self.log.error(f'Autofocus period must be at least {self._MIN_PERIOD:.0f} s.')
            else:
                self._period = float(period)
                if self._enabled and self._run is None:
                    self._restart_period_timer()
        self._emit_status()

    def _restart_period_timer(self):
        self._period_timer.start(int(round(self._period * 1000)))
        self._next_due = time.time() + self._period

    def _period_elapsed(self):
        with self._lock:
            enabled = self._enabled
        if enabled:
            self._trigger('timer')

    # ------------------------------------------------------------------ autofocus sequence
    def _trigger(self, reason):
        with self._lock:
            if self._run is not None:
                self.log.info(f'Autofocus request ({reason}) ignored: an autofocus is already running.')
                return
            self._period_timer.stop()
            self._next_due = None
            now = time.time()
            self._run = {'reason': reason, 't_trigger': now, 'phase': '', 'phase_t0': now,
                         'paused_pulsed': False, 'stopped_odmr': False, 'cw_off': False, 'paused_confirmed': False,
                         'resume_sent': False, 'resumed': list(), 'deferred': list(), 'start_pos': None,
                         'optimizer_done': False, 'optimizer_seen': False, 'refused': 0, 'optimal': dict(),
                         'fit_failed': False, 'fit_r2': list(), 'timed_out': False, 'cancelled': False,
                         'interfered': False, 'result': None, 'note': ''}
            self._set_phase('waiting_for_scan', 'waiting for a running scan to finish')
            self._poll_timer.start()
        self.log.info(f'Autofocus requested ({reason}).')

    def _cancel(self):
        with self._lock:
            run = self._run
            if run is None or run['cancelled']:
                return
            if run['phase'] == 'resuming':
                # the result is already applied (scanner moved, POI updated); nothing left to cancel
                self.log.info('Autofocus cancel ignored: already resuming the paused measurements.')
                return
            run['cancelled'] = True
            run['result'] = 'cancelled'
            self._add_note(run, 'cancelled by the operator')
            self.log.info('Autofocus cancelled by the operator.')
            phase = run['phase']
            try:
                if phase == 'waiting_for_scan':
                    self._finish()
                elif phase == 'waiting_paused':
                    self._resume_others()
                elif phase == 'focusing':
                    self._set_phase('stopping_optimizer', 'stopping the optimizer (cancelled)')
                    self._sigStopOptimize.emit()
                # pausing: the poll resumes once the queued pause commands have taken effect;
                # stopping_optimizer / resuming: the sequence ends by itself
            except Exception:
                self._handle_exception()

    def _poll(self):
        with self._lock:
            run = self._run
            if run is None:
                self._poll_timer.stop()
                return
            phase = run['phase']
            elapsed = time.time() - run['phase_t0']
            try:
                if phase == 'waiting_for_scan':
                    if self._poi_track_on():
                        run['result'] = 'failed'
                        self._add_note(run, 'POI Manager periodic refocus (Track) is on; switch it off - '
                                            'the autofocus replaces it')
                        self.log.error(f'Autofocus: {run["note"]}.')
                        self._finish()
                    elif self._blocker() is None:
                        self._pause_others()
                elif phase == 'pausing':
                    if self._others_paused(run):
                        run['paused_confirmed'] = True
                        if run['cancelled']:
                            self._resume_others()
                        elif elapsed >= self._settle_time:
                            self._start_refocus()
                    elif elapsed > self._phase_timeout:
                        if not run['cancelled']:
                            run['result'] = 'failed'
                        self._add_note(run, 'could not pause the other measurements in time; refocus skipped')
                        self.log.error(f'Autofocus: {run["note"]}.')
                        self._resume_others()
                elif phase == 'waiting_paused':
                    if self._blocker() is None:
                        self._start_refocus()
                elif phase == 'focusing':
                    optimizer_state = self._optimize_logic().module_state()
                    if optimizer_state == 'locked':
                        run['optimizer_seen'] = True
                    intruder = self._intruder(run)
                    if intruder:
                        run['interfered'] = True
                        self._add_note(run, intruder + ' during the refocus; refocus stopped')
                        self.log.error(f'Autofocus: {intruder} during the refocus; stopping the optimizer.')
                        self._set_phase('stopping_optimizer', 'stopping the optimizer (interference)')
                        self._sigStopOptimize.emit()
                    elif run['optimizer_done'] and optimizer_state == 'idle':
                        self._evaluate()
                    elif (not run['optimizer_seen'] and optimizer_state == 'idle'
                          and elapsed > self._REFUSED_AFTER):
                        # start_optimize returned early: scan logic or optimizer became busy after our check
                        run['refused'] += 1
                        if run['refused'] > self._MAX_REFUSED:
                            run['result'] = 'failed'
                            self._add_note(run, f'optimizer start refused {run["refused"]} times; refocus skipped')
                            self.log.error(f'Autofocus: {run["note"]}.')
                            self._resume_others()
                        else:
                            if run['refused'] == 1:
                                self._add_note(run, 'optimizer start refused (busy); retried')
                            self._set_phase('waiting_paused',
                                            f'measurements paused; waiting: {self._blocker() or "optimizer"}')
                    elif elapsed > self._optimize_timeout:
                        run['timed_out'] = True
                        self.log.error('Autofocus: optimizer timed out; stopping it.')
                        self._set_phase('stopping_optimizer', 'stopping the optimizer (timeout)')
                        self._sigStopOptimize.emit()
                elif phase == 'stopping_optimizer':
                    if run['optimizer_done'] and self._optimize_logic().module_state() == 'idle':
                        self._evaluate()
                    elif elapsed > self._phase_timeout:
                        self._add_note(run, 'optimizer did not stop')
                        self._evaluate()
                elif phase == 'resuming' and run['deferred']:
                    force = elapsed > self._phase_timeout
                    if force:
                        self._add_note(run, 'pause of ' + ', '.join(run['deferred'])
                                            + ' not confirmed in time; resume sent anyway - check it')
                        self.log.error(f'Autofocus: {run["note"]}.')
                    self._send_deferred(run, force=force)
                elif phase == 'resuming':
                    resumed = self._others_resumed(run)
                    if resumed or elapsed > self._phase_timeout:
                        if not resumed:
                            self._add_note(run, 'resume not confirmed in time - check the paused measurements')
                            self.log.error(f'Autofocus: {run["note"]}.')
                        self._finish()
            except Exception:
                self._handle_exception()

    def _handle_exception(self):
        run = self._run
        self.log.exception('Autofocus sequence error; resuming the other measurements.')
        run['result'] = run['result'] or 'failed'
        self._add_note(run, 'exception (see log)')
        try:
            if run['phase'] in ('focusing', 'stopping_optimizer'):
                self._sigStopOptimize.emit()
            if run['phase'] != 'resuming' and not run['resume_sent']:
                self._resume_others()
                return
        except Exception:
            self.log.exception('Autofocus: error while resuming the other measurements.')
        self._finish()

    def _blocker(self):
        """ Reason why the refocus cannot start now, or None. """
        if self._scanning_logic().module_state() != 'idle':
            return 'an image scan is running'
        if self._optimize_logic().module_state() != 'idle':
            return 'the optimizer is running (Refind / POI refocus)'
        poi_logic = self._poi_manager_logic()
        if poi_logic is not None and self._read_private(poi_logic, '_PoiManagerLogic__poi_optimization_running'):
            return 'a POI Manager refocus is pending'
        return None

    def _poi_track_on(self):
        poi_logic = self._poi_manager_logic()
        return poi_logic is not None and poi_logic.module_state() == 'locked'

    def _microwave_cw_on(self):
        microwave = self._microwave()
        if microwave is None:
            return False
        return microwave.module_state() != 'idle' and not microwave.is_scanning

    def _odmr_cw_on(self):
        """ True if the microwave runs CW with the ODMR GUI's CW parameters (switched on there). """
        microwave = self._microwave()
        odmr = self._odmr_logic()
        if microwave is None or odmr is None or not self._microwave_cw_on():
            return False
        params = odmr.cw_parameters
        return (math.isclose(microwave.cw_frequency, params['frequency'], rel_tol=0.0, abs_tol=1.0)
                and math.isclose(microwave.cw_power, params['power'], rel_tol=0.0, abs_tol=0.01))

    def _pause_others(self):
        run = self._run
        run['start_pos'] = dict(self._scanning_logic().scanner_target)
        pulsed = self._pulsed_logic()
        pulsed_locked = pulsed is not None and pulsed.module_state() == 'locked'
        if pulsed_locked and not self._read_private(pulsed, '_PulsedMeasurementLogic__is_paused'):
            run['paused_pulsed'] = True
            self._pulsed_paused = False
            self._sigPausePulsed.emit()
        odmr = self._odmr_logic()
        if odmr is not None and odmr.module_state() == 'locked':
            run['stopped_odmr'] = True
            self._sigStopOdmr.emit()
        elif not pulsed_locked and self._odmr_cw_on():
            # CW output switched on in the ODMR GUI: off through the ODMR logic (keeps its GUI in sync).
            # Any other microwave output is only noted when the refocus starts.
            run['cw_off'] = True
            self._sigOdmrCw.emit(False)
        what = [name for name, flag in (('pulsed', run['paused_pulsed']), ('ODMR', run['stopped_odmr']),
                                        ('ODMR CW output', run['cw_off'])) if flag]
        self._set_phase('pausing', 'pausing ' + ', '.join(what) if what else 'nothing to pause')

    def _others_paused(self, run):
        if run['paused_pulsed']:
            pulsed = self._pulsed_logic()
            if pulsed.module_state() != 'locked':
                # stopped by the operator meanwhile: nothing to pause, nothing to resume
                run['paused_pulsed'] = False
                self._add_note(run, 'pulsed measurement stopped by the operator; not resumed')
            elif not (self._read_private(pulsed, '_PulsedMeasurementLogic__is_paused') or self._pulsed_paused):
                return False
        if run['stopped_odmr'] and self._odmr_logic().module_state() != 'idle':
            return False
        if run['cw_off'] and self._microwave().module_state() != 'idle':
            return False
        return True

    def _start_refocus(self):
        run = self._run
        blocker = self._blocker()
        if blocker is not None:
            self._set_phase('waiting_paused', f'measurements paused; waiting: {blocker}')
            return
        if self._microwave() is not None and self._microwave().module_state() != 'idle':
            self._add_note(run, 'microwave output was on during the refocus (not controlled by the autofocus)')
            self.log.warning('Autofocus: the microwave output is on during the refocus.')
        run['optimizer_done'] = False
        run['optimizer_seen'] = False
        run['optimal'] = dict()
        run['fit_failed'] = False
        run['fit_r2'] = list()
        self._set_phase('focusing', 'optimizing position')
        self._sigStartOptimize.emit()

    def _intruder(self, run):
        """ A lower-priority measurement (re)started while the optimizer runs, or None. """
        pulsed = self._pulsed_logic()
        if (pulsed is not None and pulsed.module_state() == 'locked'
                and not self._read_private(pulsed, '_PulsedMeasurementLogic__is_paused')):
            return 'pulsed measurement continued' if run['paused_pulsed'] else 'pulsed measurement started'
        odmr = self._odmr_logic()
        if odmr is not None and odmr.module_state() == 'locked':
            return 'ODMR scan started'
        return None

    def _optimizer_state_changed(self, is_running, optimal_position=None, fit_data=None):
        with self._lock:
            run = self._run
            if run is None or run['phase'] not in ('focusing', 'stopping_optimizer'):
                return
            if is_running:
                if optimal_position:
                    run['optimal'].update(optimal_position)
                    if fit_data is None:
                        # the optimizer reports the window centre and stops after a failed fit
                        run['fit_failed'] = True
                    else:
                        # goodness of the Gaussian fit of this optimizer step (scale-free: independent of
                        # the count level, so a slowly fading sample still passes; a fit on noise does not)
                        result = fit_data.get('full_fit_res') if isinstance(fit_data, dict) else None
                        rsquared = getattr(result, 'rsquared', None)
                        if rsquared is not None:
                            run['fit_r2'].append((','.join(optimal_position), float(rsquared)))
            else:
                run['optimizer_done'] = True

    def _evaluate(self):
        run = self._run
        scan_logic = self._scanning_logic()
        start = run['start_pos'] or dict()
        end = dict(scan_logic.scanner_target)
        shift = {ax: float(end[ax] - start[ax]) for ax in start if ax in end}
        magnitude = math.sqrt(sum(v * v for v in shift.values()))
        has_position = run['optimal'] and not (run['cancelled'] or run['interfered'])
        edge_axes = self._edge_axes(start, end) if has_position else list()
        if edge_axes:
            # The optimizer clamps a fit centre outside its window to the window edge: the real
            # optimum may lie further away. Informational only (no count/position gating beyond
            # max_shift), the next autofocus continues from here.
            self._add_note(run, 'optimum at the optimizer window edge on ' + ','.join(edge_axes))
        if run['cancelled']:
            run['result'] = 'cancelled'
        elif run['interfered']:
            run['result'] = 'failed'
        elif run['timed_out']:
            run['result'] = 'failed'
            self._add_note(run, 'optimizer timed out')
        elif not run['optimal']:
            run['result'] = 'failed'
            self._add_note(run, 'optimizer returned no position')
        elif run['fit_failed']:
            run['result'] = 'failed'
            self._add_note(run, 'optimizer fit failed')
        elif self._poor_fit(run):
            run['result'] = 'failed'
            self._add_note(run, 'no clear spot in the optimizer window (fit R^2 '
                                + ', '.join(f'{axes} {r2:.2f}' for axes, r2 in run['fit_r2'])
                                + f' < {float(self._min_fit_rsquared):.2f})')
        elif self._max_shift is not None and magnitude > float(self._max_shift):
            run['result'] = 'rejected'
            self._add_note(run, f'shift {magnitude * 1e6:.3f} um exceeds max_shift '
                                f'{float(self._max_shift) * 1e6:.3f} um')
        else:
            run['result'] = 'accepted'
        end_after = end
        if run['result'] != 'accepted' and start and magnitude > 0:
            # Back to where the measurement was running; the POI list is not changed.
            if scan_logic.module_state() == 'idle':
                scan_logic.set_target_position(start, move_blocking=True)
                end_after = dict(scan_logic.scanner_target)
                self._add_note(run, 'returned to start position')
            else:
                self._add_note(run, 'could not return to the start position (scan still running)')
            self.log.warning(f'Autofocus {run["result"]}: {run["note"]}.')
        elif run['result'] == 'accepted':
            self._update_active_poi(start, end)
        run['end_pos'] = end_after
        run['shift'] = shift
        run['shift_magnitude'] = magnitude
        self._resume_others()

    def _poor_fit(self, run):
        """ True if any optimizer step's Gaussian fit has R^2 below min_fit_rsquared. """
        if not self._min_fit_rsquared or not run['fit_r2']:
            return False
        threshold = float(self._min_fit_rsquared)
        return any(not r2 >= threshold for _, r2 in run['fit_r2'])   # NaN counts as poor

    def _edge_axes(self, start, end):
        """ Axes whose final position lies at the edge of the optimizer scan window. """
        try:
            ranges = dict(self._optimize_logic().scan_range)
            constraints = self._scanning_logic().scanner_constraints
        except Exception:
            return list()
        axes = list()
        for ax, rel_range in ranges.items():
            if ax not in start or ax not in end or not rel_range > 0:
                continue
            position = constraints.axes[ax].position
            low = position.clip(start[ax] - rel_range / 2)
            high = position.clip(start[ax] + rel_range / 2)
            tolerance = 0.02 * rel_range
            if end[ax] <= low + tolerance or end[ax] >= high - tolerance:
                axes.append(ax)
        return axes

    def _update_active_poi(self, start, end):
        poi_logic = self._poi_manager_logic()
        if poi_logic is None:
            return
        try:
            name = poi_logic.active_poi
            if not name:
                return
            axes = ('x', 'y', 'z')
            anchor = np.asarray(poi_logic.get_poi_position(name), dtype=float)
            start_vec = np.array([start.get(ax, 0.0) for ax in axes])
            distance = float(np.linalg.norm(anchor - start_vec))
            if self._poi_tolerance is not None and distance > float(self._poi_tolerance):
                self._add_note(self._run, f'scanner was {distance * 1e6:.2f} um from the active POI '
                                          f'"{name}"; POI list not changed')
                return
            poi_logic.move_roi_from_poi_position(name=name,
                                                 position=np.array([end.get(ax, 0.0) for ax in axes]))
            self._add_note(self._run, f'POI "{name}" updated')
        except Exception:
            self.log.exception('Autofocus: could not update the active POI.')

    def _resume_others(self):
        run = self._run
        self._send_resume(run)
        names = {'odmr': 'ODMR', 'pulsed': 'pulsed', 'cw': 'ODMR CW output'}
        what = [names[key] for key in run['resumed'] + run['deferred']]
        self._set_phase('resuming', 'resuming ' + ', '.join(what) if what else 'nothing to resume')

    def _send_resume(self, run):
        """ Resume what this autofocus paused, unless the operator has changed it meanwhile.

        While still in 'pausing' (exception, pause timeout or deactivation before the pauses were
        confirmed) a queued pause/stop may not have run yet: the state checks below would skip the
        resume and the late pause would leave the measurement paused, and a resume sent right away
        can overtake work the pause itself queues (the pulsed pause queues its analysis-timer stop).
        So each of those resumes is deferred until its pause has taken effect (_send_deferred; polled
        in 'resuming', bounded wait in on_deactivate).
        """
        run['resume_sent'] = True
        if run['phase'] == 'pausing' and not run['paused_confirmed']:
            run['deferred'] = [key for key, flag in (('odmr', run['stopped_odmr']), ('cw', run['cw_off']),
                                                     ('pulsed', run['paused_pulsed'])) if flag]
            run['resumed'] = list()
            self._send_deferred(run)
            return
        resumed = list()
        odmr = self._odmr_logic()
        if run['stopped_odmr']:
            if odmr.module_state() == 'locked':
                self._add_note(run, 'ODMR scan restarted by the operator; not continued')
            else:
                self._sigContinueOdmr.emit()
                resumed.append('odmr')
        if run['cw_off']:
            if odmr.module_state() == 'locked' or self._microwave().module_state() != 'idle':
                self._add_note(run, 'microwave in use; ODMR CW output not switched on again')
            else:
                self._sigOdmrCw.emit(True)
                resumed.append('cw')
        if run['paused_pulsed']:
            pulsed = self._pulsed_logic()
            if pulsed.module_state() != 'locked':
                self._add_note(run, 'pulsed measurement stopped by the operator; not resumed')
            elif not self._read_private(pulsed, '_PulsedMeasurementLogic__is_paused'):
                self._add_note(run, 'pulsed measurement already continued by the operator')
            else:
                self._sigContinuePulsed.emit()
                resumed.append('pulsed')
        run['resumed'] = resumed

    def _pause_effective(self, key):
        """ True once the pause/stop this autofocus requested for <key> has run in the target module. """
        if key == 'odmr':
            return self._odmr_logic().module_state() == 'idle'
        if key == 'cw':
            return self._microwave().module_state() == 'idle'
        pulsed = self._pulsed_logic()
        return (pulsed.module_state() != 'locked'
                or self._read_private(pulsed, '_PulsedMeasurementLogic__is_paused'))

    def _send_deferred(self, run, force=False):
        """ Send each deferred resume whose pause has taken effect (every remaining one if force). """
        for key in list(run['deferred']):
            if not (force or self._pause_effective(key)):
                continue
            run['deferred'].remove(key)
            if key == 'pulsed' and self._pulsed_logic().module_state() != 'locked':
                self._add_note(run, 'pulsed measurement stopped by the operator; not resumed')
                continue
            if key == 'odmr':
                self._sigContinueOdmr.emit()
            elif key == 'cw':
                self._sigOdmrCw.emit(True)
            else:
                self._sigContinuePulsed.emit()
            run['resumed'].append(key)
        if not run['deferred']:
            run['phase_t0'] = time.time()   # the resume confirmation timeout starts now

    def _others_resumed(self, run):
        if 'odmr' in run['resumed'] and self._odmr_logic().module_state() != 'locked':
            return False
        if 'cw' in run['resumed'] and not self._microwave_cw_on():
            return False
        if 'pulsed' in run['resumed']:
            pulsed = self._pulsed_logic()
            if pulsed.module_state() != 'locked' or self._read_private(pulsed, '_PulsedMeasurementLogic__is_paused'):
                return False
        return True

    def _finish(self):
        run = self._run
        self._poll_timer.stop()
        run['result'] = run['result'] or 'failed'
        try:
            self._complete_positions(run)
        except Exception:
            self.log.exception('Autofocus: could not read the scanner position.')
        self._record(run)
        duration = time.time() - run['t_trigger']
        self.log.info(f'Autofocus finished ({run["reason"]}): {run["result"]}, '
                      f'shift {run.get("shift_magnitude", 0.0) * 1e6:.3f} um, {duration:.1f} s. {run["note"]}')
        self._run = None
        self._set_phase('idle', f'last autofocus: {run["result"]}')
        if self._enabled:
            self._restart_period_timer()
        self._emit_status()

    def _complete_positions(self, run):
        """ Fill end position and shift for runs that ended before the evaluation. """
        if 'end_pos' in run:
            return
        end = dict(self._scanning_logic().scanner_target)
        start = run['start_pos'] or end
        shift = {ax: float(end[ax] - start[ax]) for ax in start if ax in end}
        run['end_pos'] = end
        run['shift'] = shift
        run['shift_magnitude'] = math.sqrt(sum(v * v for v in shift.values()))

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _add_note(run, text):
        run['note'] = f'{run["note"]}; {text}' if run['note'] else text

    def _link(self, signal, slot):
        signal.connect(slot, QtCore.Qt.ConnectionType.QueuedConnection)
        self._connected.append((signal, slot))

    @staticmethod
    def _read_private(module, attribute):
        """ Read a private boolean flag of another module (atomic read; False if unavailable). """
        try:
            return bool(getattr(module, attribute))
        except Exception:
            return False

    def _pulsed_status_changed(self, is_running, is_paused):
        with self._lock:
            self._pulsed_paused = bool(is_paused)

    def _set_phase(self, phase, message=''):
        if self._run is not None:
            self._run['phase'] = phase
            self._run['phase_t0'] = time.time()
        self._phase = phase
        self._message = message
        self._emit_status()

    def _emit_status(self):
        self.sigStatusChanged.emit(self.status())

    def _record(self, run):
        stamp = datetime.datetime.now()
        shift = run.get('shift') or dict()
        end_pos = run.get('end_pos') or dict()
        entry = {'time': stamp.isoformat(timespec='seconds'),
                 'reason': run['reason'],
                 'result': run['result'],
                 'dx_um': round(shift.get('x', 0.0) * 1e6, 4),
                 'dy_um': round(shift.get('y', 0.0) * 1e6, 4),
                 'dz_um': round(shift.get('z', 0.0) * 1e6, 4),
                 'x_um': round(end_pos.get('x', 0.0) * 1e6, 4),
                 'y_um': round(end_pos.get('y', 0.0) * 1e6, 4),
                 'z_um': round(end_pos.get('z', 0.0) * 1e6, 4),
                 'paused': '+'.join(n for n, f in (('pulsed', run['paused_pulsed']),
                                                   ('odmr', run['stopped_odmr']),
                                                   ('cw', run['cw_off'])) if f) or '-',
                 'duration_s': round(time.time() - run['t_trigger'], 1),
                 'fit_r2': ' '.join(f'{axes}:{r2:.3f}' for axes, r2 in run['fit_r2']) or '-',
                 'note': run['note']}
        history = list(self._history)
        history.append(entry)
        self._history = history[-self._MAX_HISTORY:]
        try:
            directory = self.module_default_data_dir
            os.makedirs(directory, exist_ok=True)
            # first log file of the day whose columns match; a module update that changed the columns
            # continues in autofocus_log_2.csv, _3, ... (never append misaligned rows to an old file)
            index = 1
            while True:
                path = os.path.join(directory, 'autofocus_log.csv' if index == 1 else f'autofocus_log_{index}.csv')
                if not os.path.exists(path):
                    break
                with open(path, newline='', encoding='utf-8') as handle:
                    if next(csv.reader(handle), []) == list(entry):
                        break
                index += 1
            new_file = not os.path.exists(path)
            with open(path, 'a', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(entry))
                if new_file:
                    writer.writeheader()
                writer.writerow(entry)
        except Exception:
            self.log.exception('Autofocus: could not append to the CSV log.')
        self.sigHistoryUpdated.emit(list(self._history))
