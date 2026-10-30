# -*- coding: utf-8 -*-
"""
Pulsed-ODMR input interfuse for lele_ODMR_conf (2026-09-10, TASK-040).

WHY THIS EXISTS AND WHY IT IS NOT THE PULSED GUI
------------------------------------------------
qudi's built-in pulsed ODMR (logic/pulsed/predefined_generate_methods/
basic_predefined_methods.py:321 generate_pulsedodmr) puts the swept frequency on the
MICROWAVE ELEMENT itself (`self._get_mw_element(..., freq=mw_freq, ...)`), i.e. it requires the
pulse generator to SYNTHESISE the microwave. On this bench the pulse generator is a Swabian
Pulse Streamer 8/2 whose `uw_x_channel` is a DIGITAL gate (physical ch4 -> ZASWA-2-50DR+
switch); the microwave itself is a fixed-frequency CW carrier from an external SMIQ06B. A
digital gate cannot carry a frequency, so that code path is unusable here without an AWG.

What this bench CAN do is exactly what CW ODMR already does: step the SMIQ through a
LIST/JUMP table with a trigger pulse on PS ch6 ('next') and count photons per frequency point.
That machinery lives in OdmrScanInput, together with a pile of hard-won lessons. So pulsed
ODMR here is NOT a new measurement architecture -- it is the same frequency-stepping scan with
a different INTRA-POINT pulse pattern. Hence: subclass, override the pattern, inherit
everything else (CountBetweenMarkers ownership, borrow-or-own, the SCAN-001 window-closure
guarantee, teardown/final-state handling, the post-stop drain contract).

PER-POINT PATTERN
-----------------
CW ODMR (base class): laser CONSTANT high, mw gate open over a long non-coherent fraction of
the point, one long count window. Power-broadened linewidth.

Pulsed ODMR (this class): laser OFF while the spin is driven, mw gate = a calibrated pi pulse,
then a short laser pulse that BOTH reads out and re-polarises, with the count window inside
it. One point = N such shots, accumulated:

    next:  [pn HIGH][rest LOW]                       (advance SMIQ to this point's frequency)
    lead:  pn + mw_settle_time                       (SMIQ settle -- ODMR-001, the real rate limiter)
    then N x shot, then a tail that pads the block to exactly T:

        shot = [gap][pi (mw HIGH)][gap][readout (laser HIGH, detect HIGH inside)][wait]

    laser:  LOW except during each readout    <- the essential difference from CW
    mw:     HIGH only during each pi pulse
    detect: HIGH for detect_ns, starting detect_delay_ns after the readout laser edge

N windows per point (one per shot) instead of the base's 1 (sum) or 2 (diff), so both
_init_cbm sizing and the window->rate conversion are overridden.

WHAT MUST BE MEASURED BEFORE THIS IS TRUSTWORTHY
------------------------------------------------
* pi_pulse_ns has NO defensible default. It comes from a Rabi measurement at the SAME
  microwave power (the reference setup measured pi = 1.05 us at -7 dBm, 466 kHz Rabi). The
  option is declared missing='error' on purpose: the module refuses to load rather than
  silently invent a pulse length, because a wrong pi does not error -- it just quietly gives a
  shallow or absent dip, which is the single hardest symptom to attribute on this system.
* detect_delay_ns / detect_ns want a scope or a photon-arrival histogram: the readout contrast
  lives in the first few hundred ns of the laser pulse, and counting into the laser turn-on
  transient dilutes it.
* The reference setup's PROBE FIRST warning applies here as well: verify empirically that the
  gating actually lands where this code believes it does before believing any dip.

SAFETY
------
Activation performs no RF and no motion. The laser only goes high inside the readout segments
and only if the inherited enable_laser / manual laser_on state allows it -- with
enable_laser False this module emits the full timing pattern with the laser gate held LOW, so
it can be probed on a scope with no light. RF-ON remains a human action at the amplifier PSU
(safety_approval_log SAFE-001, cap -7 dBm).
"""

import threading

import numpy as np

from qudi.core.configoption import ConfigOption
from qudi.hardware.interfuse import odmr_scan_input as _base   # its TimeTagger import (tt)
from qudi.hardware.interfuse.odmr_scan_input import OdmrScanInput


class PulsedOdmrScanInput(OdmrScanInput):
    """ Pulsed ODMR via SMIQ list-mode frequency stepping + a per-point pi/readout pattern.

    Example config:

        pulsed_odmr_scan_input:
            module.Class: 'interfuse.pulsed_odmr_scan_input.PulsedOdmrScanInput'
            connect:
                tagger_provider: timetagger_provider
                ps_provider: pulsestreamer_provider
            options:
                ps_channels: {laser: 0, mw: 4, detect: 1, next: 6}
                apd_channels: [1, 2]
                detect_tt_channel: 5
                pi_pulse_ns: 1050          # FROM RABI -- no default
                readout_ns: 400
                detect_ns: 300
                detect_delay_ns: 20
                gap_ns: 100
                wait_ns: 1000
                enable_laser: False
                # optional normalised readout (2026-10-23): scan_mode 'norm' adds a reference
                # window late in the SAME readout laser pulse; pair with OdmrLogic
                # reference_channel '<input>_ref' for a baseline-1 contrast plot.
                # scan_mode: 'norm'
                # ref_delay_ns: 2604       # inside readout_ns, after the detect window
                # ref_ns: 369

    Runtime: set_pulse_params(pi_pulse_ns=..., detect_delay_ns=..., ...) between sweeps; the
    per-point sequence is rebuilt at every frame start.

    Time trace (2026-10-23): every frame also runs a TimeDifferences photon-arrival histogram
    (start = next = detect rising edge, one row per detect window, clicks delayed by
    detect_delay_ns + trace_pre_ns so the trace starts trace_pre_ns BEFORE the readout laser
    edge). It is summed over all shots and accumulated over sweeps until the shot geometry
    changes or reset_time_trace() is called; get_time_trace() returns it on a time axis in
    Pulse Streamer ns relative to the readout laser edge. A trace failure is logged and never
    stops the ODMR scan.
        trace_binwidth_ns: 2
        trace_pre_ns: 300
        trace_post_ns: 300
    """

    # -- pulsed shot geometry -------------------------------------------------------------
    # No default for pi: see the module docstring. missing='error' makes a missing value a
    # load failure instead of a silent wrong answer.
    _pi_pulse_ns = ConfigOption(name='pi_pulse_ns', missing='error')
    _readout_ns = ConfigOption(name='readout_ns', default=400)
    _detect_ns = ConfigOption(name='detect_ns', default=None)
    _detect_delay_ns = ConfigOption(name='detect_delay_ns', default=20)
    _gap_ns = ConfigOption(name='gap_ns', default=100)
    _wait_ns = ConfigOption(name='wait_ns', default=1000)
    # None = fit as many shots as the point period allows (the sample rate then sets the
    # per-point averaging, mirroring how the base class derives its window from T).
    _shots_per_point = ConfigOption(name='shots_per_point', default=None)
    # norm mode only: reference window inside the same readout laser pulse, on its flat part
    # (the spin-dependent transient and the laser turn-on artefacts are over by then).
    _ref_delay_ns = ConfigOption(name='ref_delay_ns', default=None)
    _ref_ns = ConfigOption(name='ref_ns', default=None)

    # laser-pulse time trace (photon-arrival histogram, see the class docstring)
    _trace_binwidth_ns = ConfigOption(name='trace_binwidth_ns', default=2)
    _trace_pre_ns = ConfigOption(name='trace_pre_ns', default=300)
    _trace_post_ns = ConfigOption(name='trace_post_ns', default=300)

    # software analysis windows (2026-10-24, scan_mode 'norm' only): the detect gate covers the
    # whole readout laser pulse, a per-point marker (ps_channels 'marker' -> marker_tt_channel)
    # OPENS each frequency point, and a per-point photon-arrival histogram is recorded. The
    # readout/reference windows (detect_delay_ns/detect_ns, ref_delay_ns/ref_ns; ns after the
    # readout laser gate edge) are then applied in software and may change at any time
    # (set_analysis_windows); PulsedOdmrLogic re-analyses all sweeps with them.
    _software_windows = ConfigOption(name='software_windows', default=False)
    _marker_tt_channel = ConfigOption(name='marker_tt_channel', default=8)
    _MARKER_NS = 20

    _PULSE_PARAMS = ('pi_pulse_ns', 'readout_ns', 'detect_ns', 'detect_delay_ns', 'gap_ns',
                     'wait_ns', 'ref_delay_ns', 'ref_ns')
    _WINDOW_PARAMS = ('detect_delay_ns', 'detect_ns', 'ref_delay_ns', 'ref_ns')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._shots = 1          # resolved shots per point
        self._wps = 1            # windows per shot: 1 (sum) | 2 (norm: readout + reference)
        self._detect_len = 0     # resolved detect window length (ns)
        self._ref_len = 0        # resolved reference window length (ns, norm only)
        self._trace_meas = None      # TimeTagger TimeDifferences (alive during a frame)
        self._trace_delayed = None   # TimeTagger DelayedChannel on the click channel
        self._trace_accum = None     # (rows, bins) over finished frames, readout row FIRST
        self._trace_key = None       # geometry the accumulator belongs to
        # Separate from _thread_lock, which the base acquire_frame holds for a whole sweep
        # line: the GUI reads the trace every second and must never wait for a line to end.
        self._trace_lock = threading.Lock()
        self._trace_skip_frame = False   # reset during a frame: discard that frame's data
        self._trace_last_row = 0         # raw row that held the readout starts last time
        self._sw = False                 # software analysis windows active (set on activation)
        self._ph_frame = None            # (points, bins) histogram of the last finished frame
        self._ph_pending = None          # finished frame waiting for pop_frame_histogram()
        self._ph_marker = None           # TimeTagger Countrate on the marker channel (row check)

    # -- lifecycle -----------------------------------------------------------------------

    def on_activate(self):
        # Validate BEFORE the base class acquires the Time Tagger / Pulse Streamer: an owned
        # tagger would otherwise leak on a config error (qudi does not deactivate a module
        # whose activation failed). Values are normalised to int ns once, here.
        if self._scan_mode not in ('sum', 'norm'):
            raise ValueError(
                f"pulsed_odmr_scan_input supports scan_mode 'sum' or 'norm' (got "
                f"'{self._scan_mode}'). The base class's 'diff' splits ONE point into an "
                f"mw-on and an mw-off half of a continuous gate, which has no meaning for a "
                f"pi pulse. 'norm' here is a readout window plus a reference window late in "
                f"the same readout laser pulse.")
        self._sw = bool(self._software_windows)
        if self._sw:
            if self._scan_mode != 'norm':
                raise ValueError("software_windows needs scan_mode 'norm' (readout + reference "
                                 "channels).")
            if 'marker' not in self._ps_channels:
                raise ValueError("software_windows needs ps_channels['marker'] (the per-point "
                                 "marker output wired to marker_tt_channel).")
        # hardware detect windows per shot: software mode gates the whole readout pulse once
        self._wps = 1 if self._sw else (2 if self._scan_mode == 'norm' else 1)
        for name, val in self._current_params().items():
            setattr(self, f'_{name}', None if val is None else int(val))
        self._detect_len, self._ref_len = self._validate_geometry(self._current_params())

        super().on_activate()
        try:
            self._resolve_shots()   # needs the sample rate the base class just set
        except Exception:
            super().on_deactivate()
            raise
        ref = (f'; reference {self._ref_len} ns at +{self._ref_delay_ns} ns'
               if self._wps == 2 else '')
        self.log.info(
            f'Pulsed ODMR ({self._scan_mode}): shot = {self._shot_length()} ns '
            f'(gap {self._gap_ns} + pi {self._pi_pulse_ns} + gap {self._gap_ns} + readout '
            f'{self._readout_ns} + wait {self._wait_ns}); detect {self._detect_len} ns at '
            f'+{self._detect_delay_ns} ns into the readout{ref}; {self._shots} shots/point at '
            f'{self._sample_rate:.3g} Hz. pi came from the config, NOT from a measurement '
            f'made by this module -- confirm it against a Rabi at the same power.')

    # -- pulse parameters ----------------------------------------------------------------

    def _current_params(self):
        return {name: getattr(self, f'_{name}') for name in self._PULSE_PARAMS}

    def _validate_geometry(self, p):
        """ Check one parameter set; return (detect_len, ref_len) in ns or raise. Shared by
        activation and set_pulse_params so a runtime change obeys the same rules. """
        for name in ('pi_pulse_ns', 'readout_ns', 'gap_ns', 'wait_ns'):
            if int(p[name]) < 1:
                raise ValueError(f'{name} must be >= 1 ns (got {p[name]}).')
        readout, delay = int(p['readout_ns']), int(p['detect_delay_ns'])
        if delay < 0:
            raise ValueError(f'detect_delay_ns must be >= 0 (got {delay}).')
        detect = readout - delay if p['detect_ns'] is None else int(p['detect_ns'])
        if detect < 1:
            raise ValueError(
                f'detect window is {detect} ns: detect_delay_ns ({delay}) must leave room '
                f'inside readout_ns ({readout}).')
        if delay + detect > readout:
            raise ValueError(
                f'detect window ends outside the readout laser pulse '
                f'({delay} + {detect} > {readout} ns). Counting after the laser turns off '
                f'measures dark counts, not contrast.')
        if self._sw:
            # software windows: both inside the readout pulse; they may overlap or be in any
            # order (no detect edges involved)
            if p['ref_delay_ns'] is None or p['ref_ns'] is None:
                raise ValueError('software windows need ref_delay_ns and ref_ns.')
            ref_delay, ref = int(p['ref_delay_ns']), int(p['ref_ns'])
            if ref < 1 or ref_delay < 0 or ref_delay + ref > readout:
                raise ValueError(f'reference window +{ref_delay}/{ref} ns must lie inside the '
                                 f'{readout} ns readout laser pulse.')
            # the marker opens the next point's row: photons of a point's last shot must be
            # out of the histogram range (pre + readout + post) before the next point starts
            # (histogram range rounded up to whole bins, as _ph_geometry builds it)
            bw = max(1, int(self._trace_binwidth_ns))
            pre, post = max(0, int(self._trace_pre_ns)), max(0, int(self._trace_post_ns))
            span = int(np.ceil((pre + readout + post) / bw)) * bw - readout
            if int(p['wait_ns']) < span:
                raise ValueError(f'software windows need wait_ns >= trace_pre_ns + '
                                 f'trace_post_ns rounded up to whole bins ({span} ns) so no '
                                 f'photon is counted in the next point.')
            return detect, ref
        if self._wps == 1:
            return detect, 0
        if p['ref_delay_ns'] is None or p['ref_ns'] is None:
            raise ValueError("scan_mode 'norm' needs ref_delay_ns and ref_ns.")
        ref_delay, ref = int(p['ref_delay_ns']), int(p['ref_ns'])
        if ref < 1:
            raise ValueError(f'ref_ns must be >= 1 ns (got {ref}).')
        # >= 1 ns LOW between the two windows: CountBetweenMarkers needs a falling and a
        # rising detect edge to separate them.
        if ref_delay < delay + detect + 1:
            raise ValueError(
                f'reference window (+{ref_delay} ns) must start after the detect window ends '
                f'(+{delay + detect} ns) with at least 1 ns gap.')
        if ref_delay + ref > readout:
            raise ValueError(
                f'reference window ends outside the readout laser pulse '
                f'({ref_delay} + {ref} > {readout} ns).')
        return detect, ref

    @property
    def pulse_params(self):
        """ Current shot geometry in ns (plus resolved window lengths and shots/point). """
        p = self._current_params()
        p.update(detect_len_ns=self._detect_len, ref_len_ns=self._ref_len,
                 shots_per_point=self._shots, shot_ns=self._shot_length())
        return p

    def set_pulse_params(self, **kwargs):
        """ Change pi/readout/window timing between sweeps (e.g. after a new Rabi). Takes
        effect at the next frame start, where the per-point sequence is rebuilt. Refused
        while a frame is running; on an invalid set nothing changes. """
        unknown = set(kwargs) - set(self._PULSE_PARAMS)
        if unknown:
            raise ValueError(f'unknown pulse parameter(s) {sorted(unknown)}; '
                             f'allowed {self._PULSE_PARAMS}.')
        with self._thread_lock:
            if self.is_running:
                raise RuntimeError('Cannot change pulse parameters while acquisition is running.')
            old = self._current_params()
            new = dict(old)
            new.update({k: (None if v is None else int(v)) for k, v in kwargs.items()})
            detect, ref = self._validate_geometry(new)
            for name, val in new.items():
                setattr(self, f'_{name}', val)
            self._detect_len, self._ref_len = detect, ref
            try:
                self._resolve_shots()
            except ValueError:
                for name, val in old.items():
                    setattr(self, f'_{name}', val)
                self._detect_len, self._ref_len = self._validate_geometry(old)
                self._resolve_shots()
                raise
        self.log.info(f'Pulsed ODMR pulse parameters now {self.pulse_params}.')

    # -- shot geometry helpers ------------------------------------------------------------

    def _shot_length(self):
        return int(self._gap_ns + self._pi_pulse_ns + self._gap_ns
                   + self._readout_ns + self._wait_ns)

    def _point_budget(self):
        """ (T, pn, lead, usable) in ns for the current sample rate, matching the base
        class's advance_at_start semantics. """
        T = int(round(1e9 / self._sample_rate))
        pn = int(max(1, min(int(self._next_pulse_ns), max(1, T // 100))))
        settle = int(round(self._mw_settle_time * 1e9))
        lead = pn + settle if self._advance_at_start else settle
        tail_pn = 0 if self._advance_at_start else pn
        # keep at least 1 ns of tail so the final detect edge closes inside this block
        # (SCAN-001) whatever the Pulse Streamer's final state is
        usable = T - lead - tail_pn - 1
        return T, pn, lead, usable

    def _resolve_shots(self):
        """ Decide how many shots fit in one point, and refuse rather than silently truncate.
        Must run BEFORE _init_cbm, because the CountBetweenMarkers is sized from
        windows_per_point (odmr_scan_input._init_cbm). """
        T, _pn, lead, usable = self._point_budget()
        shot = self._shot_length()
        if self._sw:
            # the point marker (block start) must end before the point's first readout edge;
            # checked here so set_pulse_params / set_sample_rate refuse (and roll back) at once
            first_edge = lead + self._gap_ns + self._pi_pulse_ns + self._gap_ns
            if first_edge <= self._MARKER_NS:
                raise ValueError(f'The {self._MARKER_NS} ns point marker must end before the '
                                 f'first readout ({first_edge} ns into the point): lengthen '
                                 f'mw_settle_time, gap_ns or pi_pulse_ns.')
        if usable < shot:
            raise ValueError(
                f'One pulsed shot is {shot} ns but only {usable} ns are usable inside a '
                f'{T} ns point (sample_rate {self._sample_rate:.3g} Hz, mw_settle_time '
                f'{self._mw_settle_time:.3g} s). Lower the data rate, shorten the settle '
                f'time, or shorten the shot.')
        fit = usable // shot
        if self._shots_per_point is None:
            self._shots = int(fit)
        else:
            want = int(self._shots_per_point)
            if want < 1:
                raise ValueError('shots_per_point must be >= 1.')
            if want > fit:
                raise ValueError(
                    f'shots_per_point={want} does not fit: only {fit} shots of {shot} ns fit '
                    f'in the {usable} ns usable part of a {T} ns point.')
            self._shots = want
        self._windows_per_point = self._shots * self._wps
        # per-window integration time for the c/s normalisation
        self._win_time = self._detect_len * 1e-9

    def set_sample_rate(self, rate):
        """ Re-resolve the shot count: changing the rate changes how many shots fit. If the
        shot no longer fits, the previous rate is restored before the error propagates. """
        old_rate = self._sample_rate
        super().set_sample_rate(rate)
        try:
            self._resolve_shots()
        except ValueError:
            self._sample_rate = old_rate
            self._resolve_shots()
            raise

    def set_scan_mode(self, mode):
        """ The base class would set windows_per_point to 1/2 here and break the shot
        accounting. The channel set (sum: 1, norm: 2) is fixed at activation, so only the
        configured mode is accepted; change scan_mode in the config and reload instead. """
        if str(mode) != self._scan_mode:
            raise ValueError(f"pulsed_odmr_scan_input runs scan_mode '{self._scan_mode}' "
                             f"(fixed at activation); change the config and reload.")
        self._windows_per_point = self._shots * self._wps

    def _init_cbm(self):
        """ Re-resolve shots first so the CountBetweenMarkers gets n_values =
        shots * frame_size, then let the base class build it. The time-trace measurement is
        created here too, so the base class's arm barrier (SCAN-006) covers it. """
        self._resolve_shots()
        super()._init_cbm()
        if self._sw:
            # the per-point histogram IS the data in software mode: no fallback, fail loudly
            self._start_point_histograms()
            return
        try:
            self._start_trace()
        except Exception:
            self._trace_meas, self._trace_delayed = None, None
            self.log.exception('Pulsed ODMR time trace could not be started; the scan continues '
                               'without it:')

    def _teardown(self):
        """ Fold the frame's time trace into the accumulator before the base class releases
        the Time Tagger measurements. Called with the thread lock held. """
        try:
            if self._sw:
                self._collect_point_histograms()
            else:
                self._collect_trace()
        except Exception:
            self.log.exception('Could not collect the pulsed ODMR time trace:')
        super()._teardown()

    # -- software windows: per-point histograms ------------------------------------------

    def _ph_geometry(self):
        """ (binwidth_ns, n_bins, pre_ns, click_delay_ns, key): time zero = readout laser gate
        edge (the detect rising edge), trace_pre_ns shown before it via delayed clicks. """
        bw = max(1, int(self._trace_binwidth_ns))
        pre, post = max(0, int(self._trace_pre_ns)), max(0, int(self._trace_post_ns))
        n_bins = int(np.ceil((pre + int(self._readout_ns) + post) / bw))
        return bw, n_bins, pre, pre, ('sw', bw, n_bins, pre, int(self._readout_ns))

    def _start_point_histograms(self):
        """ One histogram row per frequency point: start = detect rising edge (each shot's
        readout laser), next = per-point marker at the START of each point. TimeDifferences
        only starts with the first next_channel tag (Time Tagger manual), so the first marker
        opens row 0 and point k fills row k; no rollover inside a frame. (Fix 2026-10-24: a
        marker after each point lost point 0 and shifted the spectrum by one step.) Created
        inside the base arm barrier (SCAN-006). """
        bw, n_bins, _pre, delay, key = self._ph_geometry()
        click = self._combiner.getChannel() if self._combiner is not None \
            else int(self._apd_channels[0])
        delayed = _base.tt.DelayedChannel(self._tagger, click, int(delay * 1000))
        meas = _base.tt.TimeDifferences(self._tagger,
                                        click_channel=delayed.getChannel(),
                                        start_channel=int(self._detect_tt_channel),
                                        next_channel=int(self._marker_tt_channel),
                                        binwidth=int(bw * 1000),
                                        n_bins=n_bins,
                                        n_histograms=int(self._frame_size))
        # counts the marker edges: point k lands in row k only if every marker arrived
        marker = _base.tt.Countrate(self._tagger, [int(self._marker_tt_channel)])
        with self._trace_lock:
            if self._trace_key != key or self._trace_accum is None:
                self._trace_accum = np.zeros((1, n_bins), dtype=np.int64)
                self._trace_key = key
            self._trace_delayed, self._trace_meas = delayed, meas
            self._trace_skip_frame = False
            self._ph_frame = None
            self._ph_marker = marker

    def _ph_current(self):
        """ (points, bins) of the running frame, or of the last finished one. """
        meas = self._trace_meas
        if meas is not None:
            try:
                return np.asarray(meas.getData(), dtype=np.int64)
            except Exception:
                pass
        return self._ph_frame

    def _frame_complete(self, markers, data=None):
        """ True when every shot of every point closed (CountBetweenMarkers), exactly one
        marker per point arrived and the first and last rows hold data. A timed-out or
        aborted frame, a lost/extra marker, or a first marker the histogram missed (rows
        shifted by one frequency step, an empty first or last row) is not handed to the
        logic. """
        if self._timed_out or self._cbm is None:
            return False
        closed = int(np.count_nonzero(self._cbm.getBinWidths()))
        if closed < self._shots * self._frame_size:
            return False
        if markers != self._frame_size:
            self.log.warning(f'Pulsed ODMR: {markers} point markers on TT ch'
                             f'{self._marker_tt_channel} for {self._frame_size} points; this '
                             f'sweep is not used (histogram rows would be shifted). Check the '
                             f"PS ch{self._ps_channels['marker']} -> TT ch"
                             f'{self._marker_tt_channel} cable and trigger level.')
            return False
        if data is not None:
            rows = np.asarray(data).sum(axis=1)
            # with real light every row has counts; an empty end row means shifted rows
            if (rows.size != self._frame_size
                    or (rows.sum() >= 20 * rows.size and (rows[0] == 0 or rows[-1] == 0))):
                self.log.warning(f'Pulsed ODMR: histogram row 0 or {rows.size - 1} is empty '
                                 f'while the others hold counts; this sweep is not used '
                                 f'(rows shifted by one point).')
                return False
        return True

    def _collect_point_histograms(self):
        meas, marker = self._trace_meas, self._ph_marker
        if meas is None:
            return
        try:
            try:
                # process the last delayed clicks (up to trace_pre after the final window)
                self._tagger.sync()
            except Exception:
                pass
            data = np.asarray(meas.getData(), dtype=np.int64)
            markers = int(np.asarray(marker.getCountsTotal())[0]) if marker is not None else -1
            complete = self._frame_complete(markers, data)
            bw, _n, pre, _d, _k = self._ph_geometry()
            with self._trace_lock:
                self._ph_frame = data
                self._ph_pending = dict(
                    hist=data, shots=int(self._shots), binwidth_ns=bw, pre_ns=pre,
                    readout_ns=int(self._readout_ns),
                    pulse=(int(self._pi_pulse_ns), int(self._gap_ns), int(self._wait_ns))
                ) if complete else None
                if (not self._trace_skip_frame and self._trace_accum is not None
                        and data.shape[1] == self._trace_accum.shape[1]):
                    self._trace_accum[0] += data.sum(axis=0)
        finally:
            for m in (meas, marker):
                try:
                    if m is not None:
                        m.stop()
                except Exception:
                    pass
            with self._trace_lock:
                self._trace_meas, self._trace_delayed, self._ph_marker = None, None, None
                self._trace_skip_frame = False

    def pop_frame_histogram(self):
        """ The last finished frame's per-point histograms (dict: hist (points, bins), shots,
        binwidth_ns, pre_ns, readout_ns), handed out once; None if there is none. """
        with self._trace_lock:
            frame, self._ph_pending = self._ph_pending, None
        return frame

    @property
    def analysis_windows(self):
        """ Software windows in ns after the readout laser gate edge, plus the histogram axis
        parameters needed to apply them. """
        bw, _n, pre, _d, _k = self._ph_geometry()
        with self._trace_lock:
            return dict(detect_delay_ns=self._detect_delay_ns, detect_ns=self._detect_len,
                        ref_delay_ns=self._ref_delay_ns, ref_ns=self._ref_len,
                        binwidth_ns=bw, pre_ns=pre, readout_ns=int(self._readout_ns),
                        software=self._sw)

    def set_analysis_windows(self, **kwargs):
        """ Change the readout/reference software windows at any time, also during a sweep
        (software_windows only). They only change how recorded photons are counted. """
        if not self._sw:
            raise RuntimeError('Analysis windows are hardware gates unless software_windows is '
                               'enabled; use set_pulse_params between sweeps.')
        unknown = set(kwargs) - set(self._WINDOW_PARAMS)
        if unknown:
            raise ValueError(f'unknown window parameter(s) {sorted(unknown)}; allowed '
                             f'{self._WINDOW_PARAMS}.')
        with self._trace_lock:
            new = self._current_params()
            new.update({k: int(round(v)) for k, v in kwargs.items()})
            detect, ref = self._validate_geometry(new)
            for name in self._WINDOW_PARAMS:
                setattr(self, f'_{name}', new[name])
            self._detect_len, self._ref_len = detect, ref
        return self.analysis_windows

    # -- laser-pulse time trace ----------------------------------------------------------

    def _trace_geometry(self):
        """ (binwidth_ns, n_bins, pre_ns, click_delay_ns, key) of the time trace. """
        bw = max(1, int(self._trace_binwidth_ns))
        pre, post = max(0, int(self._trace_pre_ns)), max(0, int(self._trace_post_ns))
        n_bins = int(np.ceil((pre + int(self._readout_ns) + post) / bw))
        delay = int(self._detect_delay_ns) + pre
        key = (bw, n_bins, pre, int(self._detect_delay_ns), int(self._readout_ns), self._wps)
        return bw, n_bins, pre, delay, key

    def _start_trace(self):
        """ TimeDifferences with start = next = detect rising edge: in norm mode the two
        windows of a shot alternate between rows 0 and 1 (rollover to row 0 without a sync
        channel). Clicks are delayed so time zero of a row lies trace_pre_ns before the laser
        edge of that window's readout pulse. Rows are not truncated by the next start, so the
        readout row covers the whole laser pulse. """
        bw, n_bins, _pre, delay, key = self._trace_geometry()
        if self._combiner is not None:
            click = self._combiner.getChannel()
        else:
            click = int(self._apd_channels[0])
        det = int(self._detect_tt_channel)
        delayed = _base.tt.DelayedChannel(self._tagger, click, int(delay * 1000))
        meas = _base.tt.TimeDifferences(self._tagger,
                                        click_channel=delayed.getChannel(),
                                        start_channel=det,
                                        next_channel=det,
                                        binwidth=int(bw * 1000),
                                        n_bins=n_bins,
                                        n_histograms=self._wps)
        with self._trace_lock:
            if self._trace_key != key or self._trace_accum is None:
                self._trace_accum = np.zeros((self._wps, n_bins), dtype=np.int64)
                self._trace_key = key
            self._trace_delayed, self._trace_meas = delayed, meas
            self._trace_skip_frame = False

    def _aligned(self, data):
        """ One frame's rows with the readout row first. Without a sync channel the
        edge -> row mapping may differ between frames, so it is decided per frame. """
        row = self._readout_row(data)
        self._trace_last_row = row
        return np.roll(data, -row, axis=0) if row else data

    def _collect_trace(self):
        meas = self._trace_meas
        if meas is None:
            return
        try:
            data = np.asarray(meas.getData(), dtype=np.int64)
            with self._trace_lock:
                if (not self._trace_skip_frame and self._trace_accum is not None
                        and data.shape == self._trace_accum.shape):
                    self._trace_accum += self._aligned(data)
        finally:
            try:
                meas.stop()
            except Exception:
                pass
            with self._trace_lock:
                self._trace_meas, self._trace_delayed = None, None
                self._trace_skip_frame = False

    def _readout_row(self, rows):
        """ Which row holds the readout-window starts: the readout row is dark before the
        laser edge (time < 0) and bright after it; the reference row is bright at its start
        (it begins inside the same laser pulse). Row order alone is not trusted. """
        if rows.shape[0] == 1:
            return 0
        bw, _n, pre, _d, _k = self._trace_geometry()
        dark = slice(0, max(1, (pre // 2) // bw))
        lit = slice((pre + 200) // bw, (pre + 600) // bw)
        score = [r[dark].mean() / max(r[lit].mean(), 1e-9) for r in rows.astype(float)]
        return int(np.argmin(score))

    def get_time_trace(self):
        """ Accumulated laser-pulse time trace (finished frames + the running one), read
        without _thread_lock so a running sweep line never blocks the caller.
        Returns dict: t_ns (bin start, Pulse Streamer ns after the readout laser edge),
        counts (readout row), rows (readout row first), readout_row (raw row of the readout
        starts in the latest frame), binwidth_ns, and the window geometry for drawing.
        Software windows: the sum over all frequency points (one row). """
        if self._sw:
            return self._software_time_trace()
        bw, n_bins, pre, _delay, _key = self._trace_geometry()
        with self._trace_lock:
            if self._trace_accum is None or self._trace_accum.shape != (self._wps, n_bins):
                rows = np.zeros((self._wps, n_bins), dtype=np.int64)
            else:
                rows = self._trace_accum.copy()
            meas = None if self._trace_skip_frame else self._trace_meas
        if meas is not None:
            try:
                live = np.asarray(meas.getData(), dtype=np.int64)
                if live.shape == rows.shape:
                    rows = rows + self._aligned(live)
            except Exception:
                pass
        return dict(t_ns=np.arange(n_bins) * bw - pre, counts=rows[0], rows=rows,
                    readout_row=self._trace_last_row, binwidth_ns=bw,
                    readout_ns=int(self._readout_ns), detect_delay_ns=int(self._detect_delay_ns),
                    detect_ns=self._detect_len, ref_delay_ns=self._ref_delay_ns,
                    ref_ns=self._ref_len)

    def _software_time_trace(self):
        bw, n_bins, pre, _delay, _key = self._ph_geometry()
        with self._trace_lock:
            if self._trace_accum is None or self._trace_accum.shape != (1, n_bins):
                rows = np.zeros((1, n_bins), dtype=np.int64)
            else:
                rows = self._trace_accum.copy()
            meas = None if self._trace_skip_frame else self._trace_meas
            win = (self._detect_delay_ns, self._detect_len, self._ref_delay_ns, self._ref_len)
        if meas is not None:
            try:
                live = np.asarray(meas.getData(), dtype=np.int64)
                if live.ndim == 2 and live.shape[1] == n_bins:
                    rows = rows + live.sum(axis=0)
            except Exception:
                pass
        return dict(t_ns=np.arange(n_bins) * bw - pre, counts=rows[0], rows=rows, readout_row=0,
                    binwidth_ns=bw, readout_ns=int(self._readout_ns), detect_delay_ns=win[0],
                    detect_ns=win[1], ref_delay_ns=win[2], ref_ns=win[3])

    def reset_time_trace(self):
        """ Clear the accumulated trace. The running frame's data is discarded (not
        cleared on the device: clear() mid-frame might not reset the row index). """
        with self._trace_lock:
            if self._trace_accum is not None:
                self._trace_accum[:] = 0
            if self._trace_meas is not None:
                self._trace_skip_frame = True

    # -- the actual pattern ---------------------------------------------------------------

    def _build_point_sequence(self):
        """ One point block: SMIQ advance, settle, then N pi/readout shots, padded to T.

        Every channel pattern sums to exactly T. The trailing pad is >= 1 ns so the last
        detect falling edge is inside this block (SCAN-001) regardless of the Pulse Streamer
        final state -- the same guarantee the base class relies on. """
        self._resolve_shots()
        T, pn, lead, _usable = self._point_budget()
        shot = self._shot_length()
        used = lead + self._shots * shot
        pad = T - used
        if pad < 1:
            raise ValueError(f'Point block overflows: {used} ns used of {T} ns.')

        laser_high = self._laser_in_scan()   # False keeps the laser dark: timing only

        def pattern(segments):
            return [(int(d), int(lvl)) for d, lvl in segments if int(d) > 0]

        if self._advance_at_start:
            next_ch = pattern([(pn, 1), (T - pn, 0)])
        else:
            next_ch = pattern([(T - pn, 0), (pn, 1)])

        mw_seg, laser_seg, det_seg = [(lead, 0)], [(lead, 0)], [(lead, 0)]
        post_detect = self._readout_ns - self._detect_delay_ns - self._detect_len
        if self._sw:
            return self._build_software_sequence(T, pn, lead, pad, next_ch, laser_high, pattern)
        if self._wps == 2:
            # detect | LOW | reference | LOW, all inside the readout laser pulse
            to_ref = self._ref_delay_ns - self._detect_delay_ns - self._detect_len
            post_ref = self._readout_ns - self._ref_delay_ns - self._ref_len
        for _ in range(self._shots):
            # gap | pi | gap | readout | wait
            mw_seg += [(self._gap_ns, 0), (self._pi_pulse_ns, 1),
                       (self._gap_ns, 0), (self._readout_ns, 0), (self._wait_ns, 0)]
            laser_seg += [(self._gap_ns, 0), (self._pi_pulse_ns, 0),
                          (self._gap_ns, 0), (self._readout_ns, 1 if laser_high else 0),
                          (self._wait_ns, 0)]
            det_seg += [(self._gap_ns + self._pi_pulse_ns + self._gap_ns
                         + self._detect_delay_ns, 0),
                        (self._detect_len, 1)]
            if self._wps == 2:
                det_seg += [(to_ref, 0), (self._ref_len, 1), (post_ref + self._wait_ns, 0)]
            else:
                det_seg += [(post_detect + self._wait_ns, 0)]
        mw_seg.append((pad, 0))
        laser_seg.append((pad, 0))
        det_seg.append((pad, 0))

        seq = self._pulser.createSequence()
        seq.setDigital(int(self._ps_channels['next']), pattern(next_ch))
        seq.setDigital(int(self._ps_channels['detect']), pattern(det_seg))
        seq.setDigital(int(self._ps_channels['mw']), pattern(mw_seg))
        seq.setDigital(int(self._ps_channels['laser']), pattern(laser_seg))
        return seq

    def _build_software_sequence(self, T, pn, lead, pad, next_ch, laser_high, pattern):
        """ Software-window point block: the detect gate is HIGH over each whole readout laser
        pulse (its rising edge = histogram time zero, its count = shot complete), and the
        marker goes HIGH once at the start of the block, before the first shot, so the
        per-point histogram moves to this point's row before any of its photons. """
        first_edge = lead + self._gap_ns + self._pi_pulse_ns + self._gap_ns
        if first_edge <= self._MARKER_NS:
            raise ValueError(f'The {self._MARKER_NS} ns point marker must end before the first '
                             f'readout ({first_edge} ns into the point).')
        mw_seg, laser_seg, det_seg = [(lead, 0)], [(lead, 0)], [(lead, 0)]
        for _ in range(self._shots):
            mw_seg += [(self._gap_ns, 0), (self._pi_pulse_ns, 1),
                       (self._gap_ns, 0), (self._readout_ns, 0), (self._wait_ns, 0)]
            laser_seg += [(self._gap_ns, 0), (self._pi_pulse_ns, 0),
                          (self._gap_ns, 0), (self._readout_ns, 1 if laser_high else 0),
                          (self._wait_ns, 0)]
            det_seg += [(self._gap_ns + self._pi_pulse_ns + self._gap_ns, 0),
                        (self._readout_ns, 1), (self._wait_ns, 0)]
        mw_seg.append((pad, 0))
        laser_seg.append((pad, 0))
        det_seg.append((pad, 0))
        marker_seg = [(self._MARKER_NS, 1), (T - self._MARKER_NS, 0)]
        seq = self._pulser.createSequence()
        seq.setDigital(int(self._ps_channels['next']), pattern(next_ch))
        seq.setDigital(int(self._ps_channels['detect']), pattern(det_seg))
        seq.setDigital(int(self._ps_channels['mw']), pattern(mw_seg))
        seq.setDigital(int(self._ps_channels['laser']), pattern(laser_seg))
        seq.setDigital(int(self._ps_channels['marker']), pattern(marker_seg))
        return seq

    # -- window -> rate -------------------------------------------------------------------

    def _window_bins(self, start_ns, width_ns, bw, pre):
        a = int(round((start_ns + pre) / bw))
        return a, max(a + 1, int(round((start_ns + width_ns + pre) / bw)))

    def _rates_from_histograms(self, rows, shots):
        """ Readout and reference rates (c/s) per point row from (points, bins) counts summed
        over `shots` shots, with the current software windows. """
        bw, _n, pre, _d, _k = self._ph_geometry()
        with self._trace_lock:
            dd, dl, rd, rl = (self._detect_delay_ns, self._detect_len, self._ref_delay_ns,
                              self._ref_len)
        a0, a1 = self._window_bins(dd, dl, bw, pre)
        r0, r1 = self._window_bins(rd, rl, bw, pre)
        rows = np.asarray(rows, dtype=np.float64)
        ro = rows[:, a0:a1].sum(axis=1) / (shots * (a1 - a0) * bw * 1e-9)
        rf = rows[:, r0:r1].sum(axis=1) / (shots * (r1 - r0) * bw * 1e-9)
        return ro, rf

    def _points_from_windows(self, first_point, n_points):
        """ N windows per point (one per shot), summed. rate = sum(counts) / (N * t_detect).

        The base class hardcodes 1 window (sum) or 2 (diff), so this must be overridden.
        Only whole points are converted: a point whose shots are not all closed yet is not
        reported, which is what _available_points_total already guarantees via
        closed_windows // windows_per_point. Returns one array per channel, like the base
        class. Software windows: rates come from the per-point histograms instead. """
        if n_points <= 0:
            return self._empty_samples()
        if self._sw:
            hist = self._ph_current()
            if hist is None:
                return self._empty_samples()
            ro, rf = self._rates_from_histograms(hist[first_point:first_point + n_points],
                                                 self._shots)
            return {self._channel_names[0]: ro, self._channel_names[1]: rf}
        counts = np.asarray(self._cbm.getData(), dtype=np.float64)
        n, w = self._shots, self._wps
        whole = (counts.size // (n * w)) * (n * w)
        per_point = counts[:whole].reshape(-1, n, w).sum(axis=1)     # (points, window)
        sel = per_point[first_point:first_point + n_points]
        if w == 1:
            return {self._input_channel_name: sel[:, 0] / (n * self._win_time)}
        # norm: window order per shot is (readout, reference); each in its own c/s
        return {self._channel_names[0]: sel[:, 0] / (n * self._detect_len * 1e-9),
                self._channel_names[1]: sel[:, 1] / (n * self._ref_len * 1e-9)}
