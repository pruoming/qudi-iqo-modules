# -*- coding: utf-8 -*-

"""
ConfocalScanIO — custom FiniteSamplingIOInterface hardware module for the confocal_odmr setup.

One scan frame = one hardware-timed line/image acquisition coordinated across three devices
(design: Qudi_AI/setups/confocal_odmr/confocal_scanner_design.md):

  - NI DAQ (PCIe-6323): analog-output JUMP_LIST frame on ao0-2 (x/y/z piezo), sample-clocked
    EXTERNALLY on PFI0 (the NI internal clock is NOT used) — one AO sample per pixel_next edge.
    NOTE (SCAN-007): the NI AO emits its FIRST sample on the FIRST external clock edge.
  - Pulse Streamer 8/2 (master clock): streams the per-pixel sequence — pixel_next clock pulse
    (ch5 -> PFI0), detect count-gate edge pair (ch1 -> TT ch5), mw switch level (ch4), laser
    gate level (ch0).
  - Time Tagger 20: CountBetweenMarkers on the (combined) APD channels, gated by the detect
    marker (begin = rising TT ch5, end = falling TT ch5 = -ch5) -> 2 windows per pixel:
    A (mw ON half) and B (mw OFF half).

Scan modes: 'sum' = (A+B)/t_sum (normal PL), 'diff' = (A-B)/t_winA (per-pixel mw contrast;
needs the SMIQ RF on continuously — normalization still flagged open in the design doc §8).

This module sits under the stock ni_scanning_probe_interfuse (connector `scan_hardware`) and
deliberately mirrors the stock ni_x_series_finite_sampling_io method CONTRACT, incl. post-stop
drain and ValueError-on-over-request (SCAN-003/004; qudi_notes "match the stock module's FULL
contract").

Shared-device convention (qudi_notes): the Time Tagger / Pulse Streamer are borrowed from their
provider modules when the optional connectors are wired; otherwise this module opens (and then
owns) its own connection. It only asserts device output/idle states on devices it OWNS.

SAFETY: activation performs NO motion, starts NO NI task, and (if it owns the Pulse Streamer)
sets all PS outputs LOW. Laser gating (enable_laser / idle_laser_on / laser_on()) and stage
motion are HUMAN-APPROVED actions (approved_actions.yaml); RF power is capped upstream
(SAFE-005) and never touched here.

Example config (see Qudi_AI/setups/confocal_odmr/qudi_config_confocal_counter.cfg):

    confocal_scan_io:
        module.Class: 'interfuse.confocal_scan_io.ConfocalScanIO'
        connect:
            tagger_provider: timetagger_provider   # optional — borrow the shared Time Tagger
            ps_provider: pulsestreamer_provider    # optional — borrow the shared Pulse Streamer
        options:
            ni_device: 'Dev1'
            ao_channels: ['ao0', 'ao1', 'ao2']
            ao_voltage_limits: [0.0, 10.0]
            ni_sample_clock_terminal: '/Dev1/PFI0'
            pulsestreamer_ip: '169.254.8.2'   # only used when ps_provider is NOT connected
            ps_channels:
                mw: 4
                detect: 1
                pixel_next: 5
                laser: 0
            apd_channels: [1, 2]
            detect_tt_channel: 5
            input_channel_name: 'fluorescence'
            accumulate_fraction: 0.9
            pixel_settle_time: 1.0e-3
            pixel_next_at_start: True
            scan_mode: 'sum'
            enable_laser: False
            idle_laser_on: False
            default_sample_rate: 200.0
            settle_time: 0.05
            cbm_arm_delay: 0.05
            # frame_timeout: 30.0        # optional; None/omitted -> auto (2x frame + 5 s)
            # timetagger_serial: ''      # only when owning the Tagger and several attached

Cycle-2 note: this file is the BLIND REBUILD of the cycle-1 module, written fresh from the
Qudi_AI records only (rebuild_runbook.md step 21b; sources: confocal_scanner_design.md §5/§8/§10,
known_issues SCAN-001..008, connections.yaml, qudi_notes.md, stock qudi code).

This file is part of qudi. Licensed under LGPL v3.
"""

import time
import warnings

import numpy as np
import nidaqmx as ni
from nidaqmx.stream_writers import AnalogMultiChannelWriter
import pulsestreamer as pstr
import TimeTagger as tt

from qudi.core.configoption import ConfigOption
from qudi.core.connector import Connector
from qudi.interface.finite_sampling_io_interface import FiniteSamplingIOInterface, \
    FiniteSamplingIOConstraints
from qudi.util.enums import SamplingOutputMode
from qudi.util.mutex import RecursiveMutex


class ConfocalScanIO(FiniteSamplingIOInterface):
    """ FiniteSamplingIO producing one confocal scan frame by coordinating NI AO (externally
    clocked on PFI0) + Time Tagger gated counting + a Pulse Streamer per-pixel clock sequence.
    """

    # Optional providers of the shared devices (qudi_notes provider convention). If a connector
    # is not wired, this module opens its own connection and then OWNS the device.
    _tagger_provider = Connector(name='tagger_provider', interface='TimeTaggerProvider',
                                 optional=True)
    _ps_provider = Connector(name='ps_provider', interface='PulseStreamerProvider',
                             optional=True)

    # --- NI analog output ---
    _ni_device = ConfigOption(name='ni_device', default='Dev1')
    _ao_channels = ConfigOption(name='ao_channels', default=['ao0', 'ao1', 'ao2'])
    _ao_voltage_limits = ConfigOption(name='ao_voltage_limits', default=[0.0, 10.0])
    _ni_sample_clock_terminal = ConfigOption(name='ni_sample_clock_terminal',
                                             default='/Dev1/PFI0')

    # --- Pulse Streamer (master clock) ---
    _pulsestreamer_ip = ConfigOption(name='pulsestreamer_ip', default='')
    _ps_channels = ConfigOption(name='ps_channels',
                                default={'mw': 4, 'detect': 1, 'pixel_next': 5, 'laser': 0})
    # pixel_next pulse width in ns; also the SCAN-001 detect 'tail' so every count window closes
    # INSIDE its own pixel block. Kept small: window B is shorter than A by this tail (SCAN-008).
    _pixel_next_pulse_ns = ConfigOption(name='pixel_next_pulse_ns', default=100)

    # --- Time Tagger counting ---
    _apd_channels = ConfigOption(name='apd_channels', default=[1, 2])
    _detect_tt_channel = ConfigOption(name='detect_tt_channel', default=5)
    _input_channel_name = ConfigOption(name='input_channel_name', default='fluorescence')

    # --- Per-pixel sequence timing ---
    _accumulate_fraction = ConfigOption(name='accumulate_fraction', default=0.9)
    # Per-pixel settle gap BEFORE the count windows, so PL is counted after the piezo arrives
    # at the pixel position, not mid-transit (SCAN-007). Hardware-specific; 0 disables.
    _pixel_settle_time = ConfigOption(name='pixel_settle_time', default=0.0)
    # Advance the NI AO at the pixel START (trigger -> settle -> count) so pixel i is measured
    # at position p_i — the NI AO emits its first sample on the first clock edge, so advancing
    # at the pixel END measured pixel i at p_{i-1} (fixed 2-px fwd/back registration offset,
    # SCAN-007). False = legacy ordering (advance at pixel end).
    _pixel_next_at_start = ConfigOption(name='pixel_next_at_start', default=True)

    # --- Readout / lifecycle ---
    _scan_mode = ConfigOption(name='scan_mode', default='sum')  # 'sum' | 'diff'
    _enable_laser = ConfigOption(name='enable_laser', default=False)     # HUMAN-APPROVED
    _idle_laser_on = ConfigOption(name='idle_laser_on', default=False)   # HUMAN-APPROVED
    _default_sample_rate = ConfigOption(name='default_sample_rate', default=200.0)
    # SHORT post-ramp dwell before the first pixel: the interfuse ramps the AO setpoint to the
    # first scan position but starts the frame the instant the SETPOINT reaches target, with NO
    # settle wait — this fixed dwell covers the constant following-error residual (SCAN-005).
    _settle_time = ConfigOption(name='settle_time', default=0.05)
    # ARM BARRIER: a freshly created CountBetweenMarkers is not armed instantly; without
    # sync() + this delay before starting the Pulse Streamer, a variable number of first detect
    # edges are lost and the count<->position registration slips per frame (SCAN-006;
    # 0.05 s proven 12/12 at 1 kHz — do not reduce without re-probing).
    _cbm_arm_delay = ConfigOption(name='cbm_arm_delay', default=0.05)
    # Readout timeout backstop in s; None -> auto = 2 x expected frame time + 5 s (design doc
    # §6 factor — both rebuilds silently hardcoded 1.25x, a twice-recurred record gap; re-added
    # 2026-08-11, answer-check A). Raise for legitimately long frames instead of hardcoding.
    _frame_timeout = ConfigOption(name='frame_timeout', default=None)
    # Only used when this module OWNS the Time Tagger (no tagger_provider connected):
    # '' -> the only connected Tagger; set the serial when more than one is attached
    # (recurred gap, re-added 2026-08-11 — moot when borrowing, which all real configs do).
    _timetagger_serial = ConfigOption(name='timetagger_serial', default='')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._thread_lock = RecursiveMutex()

        # device handles + ownership flags (qudi_notes borrow-or-own convention)
        self._tagger = None
        self._pulser = None
        self._owns_tagger = False
        self._owns_pulser = False

        # constraints
        self._constraints = None

        # sampling state
        self._sample_rate = -1.0
        self._output_mode = SamplingOutputMode.JUMP_LIST
        self._frame_size = 0
        self._frame_buffer = None            # dict: ao channel -> 1D voltage array
        self._active_input_channels = frozenset()
        self._active_output_channels = frozenset()

        # per-frame runtime state
        self._ao_task = None
        self._cbm = None                     # TimeTagger CountBetweenMarkers
        self._combiner = None                # TimeTagger Combiner (kept alive during frame)
        self._detect_monitor = None          # TimeTagger Countrate on the detect channel
        # Manual-laser continuity (ported from the odmr sibling 2026-08-11, answer-check A:
        # a console laser_on() must survive a scan and its teardown, not be silently killed).
        self._laser_is_on = False
        self._consumed = 0                   # pixels already returned to the consumer
        self._unread = None                  # post-stop snapshot dict (SCAN-003 drain buffer)
        self._timed_out = False
        self._frame_deadline = 0.0
        self._frame_start = 0.0

        # per-window integration times, set when the sequence is built (SCAN-007/008)
        self._win_time = 0.0                 # window A duration in s (used by diff mode)
        self._sum_time = 0.0                 # window A + window B duration in s (sum mode)

    # ---------------------------------------------------------------- activation

    def on_activate(self):
        # --- config validation (SCAN-008: reject negative timing values up front — a negative
        # settle would otherwise silently deform the pulse sequence) ---
        if self._pixel_settle_time < 0 or self._settle_time < 0 or self._cbm_arm_delay < 0:
            raise ValueError('pixel_settle_time, settle_time and cbm_arm_delay must be >= 0.')
        if not 0.0 < self._accumulate_fraction < 1.0:
            raise ValueError('accumulate_fraction must be in (0, 1): the skip gap between the '
                             'two count windows is what separates their marker edges.')
        if self._scan_mode not in ('sum', 'diff'):
            raise ValueError(f'scan_mode must be "sum" or "diff", got "{self._scan_mode}".')
        required_ps_keys = {'mw', 'detect', 'pixel_next', 'laser'}
        if not required_ps_keys.issubset(self._ps_channels):
            raise ValueError(f'ps_channels must define {sorted(required_ps_keys)}.')
        if len(self._ao_voltage_limits) != 2:
            raise ValueError('ao_voltage_limits must be [min, max].')
        if int(self._pixel_next_pulse_ns) < 1:
            raise ValueError('pixel_next_pulse_ns must be >= 1 ns.')
        if self._frame_timeout is not None and not self._frame_timeout > 0:
            raise ValueError('frame_timeout must be > 0 s, or None for auto (2x frame + 5 s).')

        self._ao_channels = [str(ch).strip('/').lower() for ch in self._ao_channels]

        # --- constraints ---
        v_lim = (float(min(self._ao_voltage_limits)), float(max(self._ao_voltage_limits)))
        self._constraints = FiniteSamplingIOConstraints(
            supported_output_modes=(SamplingOutputMode.JUMP_LIST,),
            input_channel_units={self._input_channel_name: 'c/s'},
            output_channel_units={ch: 'V' for ch in self._ao_channels},
            frame_size_limits=(1, int(1e8)),
            sample_rate_limits=(0.1, 1e4),
            output_channel_limits={ch: v_lim for ch in self._ao_channels},
            # (-inf, inf) like cycle 1: diff mode legitimately produces NEGATIVE c/s. No
            # consumer enforces these limits (verified fork-wide 2026-08-09, answer-check A —
            # design doc §8 watch CLOSED); value fixed 2026-08-11 for record consistency.
            input_channel_limits={self._input_channel_name: (-np.inf, np.inf)}
        )

        # --- Time Tagger: borrow from the provider if connected, else own ---
        # (optional-connector access pattern as used by timetagger_instreamer in this fork)
        provider = None
        try:
            provider = self._tagger_provider()
        except Exception:
            provider = None
        if provider is not None:
            self._tagger = provider.get_tagger()
            self._owns_tagger = False
            self.log.info(f'Time Tagger borrowed from provider (shared connection), '
                          f'serial {self._tagger.getSerial()}.')
        else:
            if self._timetagger_serial:
                self._tagger = tt.createTimeTagger(str(self._timetagger_serial))
            else:
                self._tagger = tt.createTimeTagger()
            self._owns_tagger = True
            self.log.info(f'Time Tagger connected (owned), '
                          f'serial {self._tagger.getSerial()}.')

        # --- Pulse Streamer: borrow from the provider if connected, else own ---
        provider = None
        try:
            provider = self._ps_provider()
        except Exception:
            provider = None
        if provider is not None:
            self._pulser = provider.get_pulser()
            self._owns_pulser = False
            self.log.info('Pulse Streamer borrowed from provider (shared connection).')
        else:
            if not self._pulsestreamer_ip:
                raise ValueError('No ps_provider connected and no pulsestreamer_ip configured.')
            self._pulser = pstr.PulseStreamer(self._pulsestreamer_ip)
            self._owns_pulser = True
            self.log.info(f'Pulse Streamer connected (owned) at {self._pulsestreamer_ip}.')

        # Idle/startup output state: ONLY asserted on an OWNED Pulse Streamer — the provider
        # owns the startup state of a shared device (qudi_notes idle-state ruling).
        if self._owns_pulser:
            self._pulser.constant(self._idle_output_state())
            if self._idle_laser_on:
                self.log.warning('idle_laser_on: laser gate held HIGH while idle '
                                 '(HUMAN-APPROVED action).')
        elif self._idle_laser_on:
            self.log.warning('idle_laser_on requested but the Pulse Streamer is BORROWED — '
                             'ignored (provider owns the idle state). Use laser_on() instead.')

        # defaults
        self._sample_rate = float(self._default_sample_rate)
        self._output_mode = SamplingOutputMode.JUMP_LIST
        self._active_input_channels = frozenset({self._input_channel_name})
        self._active_output_channels = frozenset(self._ao_channels)
        self._frame_size = 0
        self._frame_buffer = None
        self._unread = None
        self._consumed = 0

    def on_deactivate(self):
        try:
            self.stop_buffered_frame()
        except Exception:
            self.log.exception('Error stopping frame on deactivate:')
        # Only reset/free devices we OWN (deferential teardown, qudi_notes).
        if self._owns_pulser and self._pulser is not None:
            try:
                self._pulser.constant(pstr.OutputState.ZERO())
            except Exception:
                pass
        if self._owns_tagger and self._tagger is not None:
            try:
                tt.freeTimeTagger(self._tagger)
            except Exception:
                pass
        self._tagger = None
        self._pulser = None

    # ---------------------------------------------------------------- properties

    @property
    def constraints(self):
        return self._constraints

    @property
    def active_channels(self):
        """ @return (frozenset, frozenset): active input channels, active output channels """
        return self._active_input_channels, self._active_output_channels

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def frame_size(self):
        return self._frame_size

    @property
    def output_mode(self):
        return self._output_mode

    @property
    def is_running(self):
        """ Read-only flag: frame acquisition running (module_state locked). """
        return self.module_state() == 'locked'

    @property
    def samples_in_buffer(self):
        """ Number of acquired but unread pixels. When not running, reports the post-stop
        snapshot length so the consumer can drain it (SCAN-003 stock contract). """
        with self._thread_lock:
            if not self.is_running:
                if self._unread is None:
                    return 0
                return len(self._unread[self._input_channel_name])
            return max(0, self._available_pixels_total() - self._consumed)

    # ---------------------------------------------------------------- setters

    def set_sample_rate(self, rate):
        assert not self.is_running, 'Unable to set sample rate while IO is running.'
        in_range, clipped = self._constraints.sample_rate_in_range(rate)
        if not in_range:
            self.log.warning(f'Sample rate {rate:.3g} Hz out of bounds '
                             f'{self._constraints.sample_rate_limits}; clipped to '
                             f'{clipped:.3g} Hz.')
        with self._thread_lock:
            self._sample_rate = float(clipped)

    def set_active_channels(self, input_channels, output_channels):
        assert not self.is_running, 'Unable to set active channels while IO is running.'
        input_channels = frozenset(self._extract_terminal(ch) for ch in input_channels)
        output_channels = frozenset(self._extract_terminal(ch) for ch in output_channels)
        assert input_channels.issubset(set(self._constraints.input_channel_names)), \
            f'Invalid input channels: ' \
            f'{input_channels.difference(self._constraints.input_channel_names)}'
        assert output_channels.issubset(set(self._constraints.output_channel_names)), \
            f'Invalid output channels: ' \
            f'{output_channels.difference(self._constraints.output_channel_names)}'
        with self._thread_lock:
            self._active_input_channels = input_channels
            self._active_output_channels = output_channels

    def set_output_mode(self, mode):
        assert not self.is_running, 'Unable to set output mode while IO is running.'
        assert self._constraints.output_mode_supported(mode), \
            f'Output mode {mode} not supported (JUMP_LIST only).'
        self._output_mode = mode

    def set_frame_data(self, data):
        assert data is None or isinstance(data, dict), \
            f'set_frame_data expects dict or None, got {type(data)}'
        assert not self.is_running, 'IO is running. Cannot set frame data.'

        if data is None:
            with self._thread_lock:
                self._frame_buffer = None
                self._frame_size = 0
            return

        data = {self._extract_terminal(ch): np.asarray(val) for ch, val in data.items()}
        assert set(data) == set(self._active_output_channels), \
            f'Keys of frame data {sorted(data)} do not match active output channels ' \
            f'{sorted(self._active_output_channels)}'
        assert self._output_mode == SamplingOutputMode.JUMP_LIST, \
            'Only JUMP_LIST output mode is supported.'
        frame_size = len(next(iter(data.values())))
        assert all(d.ndim == 1 and len(d) == frame_size for d in data.values()), \
            'Frame data values must be 1D arrays of equal length.'
        assert self._constraints.frame_size_in_range(frame_size)[0], \
            f'Frame size {frame_size} out of range.'
        for ch, arr in data.items():
            lo, hi = self._constraints.output_channel_limits[ch]
            if arr.min() < lo or arr.max() > hi:
                raise ValueError(f'Output channel {ch} voltages exceed limits [{lo}, {hi}] — '
                                 f'refusing frame (motion safety).')

        with self._thread_lock:
            self._frame_buffer = data
            self._frame_size = int(frame_size)

    # ---------------------------------------------------------------- frame lifecycle

    def start_buffered_frame(self):
        """ Arm NI AO (waiting on the external PFI0 clock), arm the Time Tagger measurement
        (with the SCAN-006 arm barrier), then start the Pulse Streamer master sequence.
        Non-blocking. """
        assert self._frame_size > 0 and self._frame_buffer is not None, \
            'No frame data set. Cannot start buffered frame.'
        assert not self.is_running, 'Frame IO already running. Cannot start.'
        assert self._constraints.sample_rate_in_range(self._sample_rate)[0], \
            f'Cannot start frame: sample rate {self._sample_rate:.3g} Hz invalid.'
        assert set(self._active_output_channels) == set(self._frame_buffer), \
            'Active output channels and frame buffer keys do not coincide.'

        self.module_state.lock()
        try:
            with self._thread_lock:
                self._consumed = 0
                self._unread = None
                self._timed_out = False

                # (1) SHORT POST-RAMP SETTLE (SCAN-005): the interfuse gives no settle wait
                # after ramping the AO setpoint to the first scan position; the piezo still
                # carries a small constant following-error residual. AO holds target
                # (nicard_ao keep_value), so a fixed dwell here is the right tool.
                if self._settle_time > 0:
                    time.sleep(self._settle_time)

                # (2) NI AO frame, externally sample-clocked on PFI0 (armed, waits for edges).
                self._init_ao_task()

                # (3) Arm the Time Tagger CountBetweenMarkers: 2 windows per pixel
                # (A = mw-on half, B = mw-off half), begin/end = detect rising/falling edge.
                self._init_cbm()
                # In-scan detect-edge monitor for the timeout diagnostic (cycle-1 capability
                # lost twice in rebuilds; re-added 2026-08-11, answer-check A / design §9):
                # discriminates "markers not reaching the TT" from "window not closing".
                try:
                    self._detect_monitor = tt.Countrate(self._tagger,
                                                        [int(self._detect_tt_channel)])
                except Exception:
                    self._detect_monitor = None
                    self.log.warning('Could not create the detect-edge Countrate monitor; '
                                     'timeout diagnostic will omit the edge rate.')

                # (4) ARM BARRIER (SCAN-006): a fresh CountBetweenMarkers needs tens of ms to
                # actually start listening. sync() is guarded so its absence cannot break
                # start-up; the fixed delay alone already fixes the arm race.
                try:
                    self._tagger.sync()
                except Exception:
                    self.log.warning('Time Tagger sync() failed/unavailable; relying on '
                                     'cbm_arm_delay alone (SCAN-006).')
                if self._cbm_arm_delay > 0:
                    time.sleep(self._cbm_arm_delay)

                # (5) Start the Pulse Streamer master sequence (immediate trigger). It emits
                # exactly frame_size repetitions of the per-pixel block and stops; every
                # detect window closes inside its own pixel block (SCAN-001 tail), and the
                # final output state is all-LOW.
                seq = self._build_pixel_sequence()
                self._pulser.setTrigger(pstr.TriggerStart.IMMEDIATE)
                self._pulser.stream(seq, self._frame_size, pstr.OutputState.ZERO())

                self._frame_start = time.time()
                frame_duration = self._frame_size / self._sample_rate
                # Timeout backstop: return data instead of hanging forever (SCAN-001).
                # Configurable per the design-§6 contract; auto = 2x + 5 s (the hardcoded
                # 1.25x both rebuilds guessed was a recurred record gap — answer-check A).
                if self._frame_timeout is None:
                    timeout = 2.0 * frame_duration + 5.0
                else:
                    timeout = float(self._frame_timeout)
                self._frame_deadline = self._frame_start + timeout
        except Exception:
            try:
                self._teardown()
            finally:
                self.module_state.unlock()
            raise

    def stop_buffered_frame(self):
        """ Abort the running frame. Snapshot unread pixels for post-stop draining
        (SCAN-003), stop PS/NI/TT, leave outputs safe. Must NOT raise if not running. """
        if not self.is_running:
            return
        with self._thread_lock:
            if not self.is_running:
                return
            self._teardown()
            self.module_state.unlock()

    def get_buffered_samples(self, number_of_samples=None):
        """ Return acquired pixels for the active input channel.

        Contract (mirrors stock ni_x_series_finite_sampling_io — the interfuse depends on it,
        SCAN-003/004):
          - not running: drain the post-stop snapshot; empty arrays once drained; ValueError
            (and ONLY ValueError) if an explicit request exceeds what is left.
          - running: an explicit request exceeding the pixels pending in the REST OF THE FRAME
            raises ValueError immediately (end-of-frame remainder, SCAN-004) — never blocks
            for pixels that can no longer arrive.
          - number_of_samples=None returns what is available (>= 1 pixel target, so the
            progressive line-by-line readout advances even with zero photons, SCAN-002).
        """
        if number_of_samples is not None:
            assert isinstance(number_of_samples, (int, np.integer)), \
                'Number of requested samples must be integer.'
            number_of_samples = int(number_of_samples)

        with self._thread_lock:
            if not self.is_running:
                return self._drain_unread(number_of_samples)
            pending_in_frame = self._frame_size - self._consumed
            if number_of_samples is not None and number_of_samples > pending_in_frame:
                raise ValueError(f'Requested {number_of_samples} samples but only '
                                 f'{pending_in_frame} pending in this frame.')
            if number_of_samples == 0 or (number_of_samples is None and pending_in_frame == 0):
                return {self._input_channel_name: np.empty(0, dtype=np.float64)}
            target = number_of_samples if number_of_samples is not None else 1

        # Wait OUTSIDE the lock so stop_buffered_frame() can always proceed.
        while True:
            with self._thread_lock:
                if not self.is_running:
                    # frame was stopped while we waited -> post-stop drain path
                    return self._drain_unread(number_of_samples)
                available = self._available_pixels_total()
                if available - self._consumed >= target:
                    n = number_of_samples if number_of_samples is not None \
                        else available - self._consumed
                    values = self._pixels_from_windows(self._consumed, n)
                    self._consumed += n
                    return {self._input_channel_name: values}
                if time.time() > self._frame_deadline and not self._timed_out:
                    self._log_timeout_diagnostic()
                    # Backstop (SCAN-001): report the frame complete; unclosed windows read
                    # as zero counts, so the consumer gets a blank remainder, not a hang.
                    self._timed_out = True
                    continue
            time.sleep(0.01)

    def get_frame(self, data=None):
        """ Blocking single-frame IO (see interface). """
        with self._thread_lock:
            if data is not None:
                self.set_frame_data(data)
            self.start_buffered_frame()
            try:
                result = self.get_buffered_samples(self._frame_size)
            finally:
                self.stop_buffered_frame()
            return result

    # ---------------------------------------------------------------- console helpers

    @property
    def laser_is_on(self):
        """ True while a laser_on() is in force (read-only; the laser is held HIGH between and during
        scans until laser_off()). Lets other modules (g2_logic) restore the operator's laser state. """
        return bool(self._laser_is_on)

    def laser_on(self):
        """ HUMAN console action (standing laser approval): hold the laser gate HIGH via a
        constant Pulse Streamer state. Works on a borrowed PS too — an explicit human call is
        the sanctioned way to light the laser outside a scan (qudi_notes). """
        if self.is_running:
            self.log.error('Refusing laser_on(): a scan frame is running.')
            return
        self._pulser.constant(self._laser_output_state(True))
        # Track it (answer-check A / design §8): a manual laser stays HIGH in-scan and is
        # restored at teardown — same behaviour as the odmr sibling's _laser_is_on.
        self._laser_is_on = True
        self.log.warning('Laser gate HIGH (console laser_on — human-approved action).')

    def laser_off(self):
        """ Return all Pulse Streamer outputs to LOW (laser off). """
        if self.is_running:
            self.log.error('Refusing laser_off(): a scan frame is running.')
            return
        self._pulser.constant(pstr.OutputState.ZERO())
        self._laser_is_on = False
        self.log.info('Pulse Streamer outputs LOW (laser off).')

    def set_pixel_settle_time(self, seconds):
        """ Runtime tuning knob (SCAN-007): per-pixel settle gap before the count windows.
        Takes effect at the next frame build. """
        if seconds < 0:
            raise ValueError('pixel_settle_time must be >= 0.')
        self._pixel_settle_time = float(seconds)

    # ---------------------------------------------------------------- internals

    @staticmethod
    def _extract_terminal(term_str):
        """ Strip device name / slashes and lowercase (stock helper behavior). """
        term = str(term_str).strip('/').lower()
        if 'dev' in term:
            term = term.split('/', 1)[-1]
        return term

    def _idle_output_state(self):
        """ Idle/teardown PS state: laser HIGH if a manual laser_on() is active (explicit
        human action — must survive a scan, answer-check A) or, on an OWNED device, iff
        idle_laser_on. All LOW otherwise. """
        if self._laser_is_on or (self._owns_pulser and self._idle_laser_on):
            return self._laser_output_state(True)
        return pstr.OutputState.ZERO()

    def _laser_output_state(self, laser_high):
        digital_high = [int(self._ps_channels['laser'])] if laser_high else []
        return pstr.OutputState(digital_high, 0.0, 0.0)

    def _init_ao_task(self):
        """ Create + start the NI AO task: JUMP_LIST frame, external sample clock on PFI0.
        The task is armed and emits one sample per pixel_next rising edge (first sample on
        the FIRST edge — SCAN-007). """
        # Anonymous task like cycle 1 (SCAN-010): a CONSTANT name + one leaked close() would
        # wedge every later scan with DAQmx "task name already exists" until a qudi restart.
        task = ni.Task()
        try:
            lo, hi = self._constraints.output_channel_limits[self._ao_channels[0]]
            for ch in self._ao_channels:
                task.ao_channels.add_ao_voltage_chan(f'/{self._ni_device}/{ch}',
                                                     min_val=lo, max_val=hi)
            task.timing.cfg_samp_clk_timing(
                rate=self._sample_rate,
                source=self._ni_sample_clock_terminal,
                active_edge=ni.constants.Edge.RISING,
                sample_mode=ni.constants.AcquisitionType.FINITE,
                samps_per_chan=self._frame_size)
            writer = AnalogMultiChannelWriter(task.out_stream)
            writer.verify_array_shape = False
            data = np.vstack([np.ascontiguousarray(self._frame_buffer[ch], dtype=np.float64)
                              for ch in self._ao_channels])
            writer.write_many_sample(data)
            task.start()
        except Exception:
            try:
                task.close()
            except Exception:
                pass
            raise
        self._ao_task = task

    def _init_cbm(self):
        """ Fresh CountBetweenMarkers on the (combined) APD channels: begin = detect rising
        edge, end = detect falling edge (same physical channel, ch_end = -ch_start —
        connections.yaml), n_values = 2 windows per pixel. """
        apd = [int(ch) for ch in self._apd_channels]
        if len(apd) > 1:
            self._combiner = tt.Combiner(self._tagger, apd)
            click_channel = self._combiner.getChannel()
        else:
            self._combiner = None
            click_channel = apd[0]
        self._cbm = tt.CountBetweenMarkers(self._tagger,
                                           click_channel=click_channel,
                                           begin_channel=int(self._detect_tt_channel),
                                           end_channel=-int(self._detect_tt_channel),
                                           n_values=2 * self._frame_size)

    def _build_pixel_sequence(self):
        """ One pixel block (repeated frame_size times by the Pulse Streamer):

        pixel_next_at_start=True (default, SCAN-007):
            [pixel_next pulse][settle gap][window A: mw ON][window B: mw OFF][tail LOW]
        legacy (False):
            [settle gap][window A: mw ON][window B: mw OFF][pixel_next pulse]

        detect is HIGH exactly during windows A and B; each window's closing (falling) edge
        falls INSIDE the pixel block so the last pixel's window B always closes (SCAN-001 —
        the tail LOW at the block end guarantees it, independent of the PS final state).
        mw is HIGH exactly during the window-A region; laser is HIGH for the whole block iff
        enable_laser. All durations in ns. """
        T = int(round(1e9 / self._sample_rate))                # pixel period
        pn = int(max(1, min(int(self._pixel_next_pulse_ns), T // 100)))
        settle = int(round(self._pixel_settle_time * 1e9))
        skip = int(round((1.0 - self._accumulate_fraction) / 2.0 * T))

        if self._pixel_next_at_start:
            usable = T - settle
        else:
            usable = T - settle - pn
        region_a = usable // 2
        region_b = usable - region_a
        win_a = region_a - skip
        tail = pn                                              # SCAN-001 / SCAN-008
        win_b = (region_b - skip - tail) if self._pixel_next_at_start else (region_b - skip)

        if win_a <= 0 or win_b <= 0:
            raise ValueError(
                f'Count windows non-positive (win_a={win_a} ns, win_b={win_b} ns) at '
                f'{self._sample_rate:.3g} Hz with pixel_settle_time='
                f'{self._pixel_settle_time:.3g} s, accumulate_fraction='
                f'{self._accumulate_fraction}: lower the sample rate or the settle time.')

        def pattern(segments):
            """ (duration_ns, level) list, dropping zero-length segments; all durations were
            validated non-negative above (SCAN-008). """
            return [(int(d), int(lvl)) for d, lvl in segments if int(d) > 0]

        if self._pixel_next_at_start:
            pixel_next = pattern([(pn, 1), (T - pn, 0)])
            detect = pattern([(settle + skip, 0), (win_a, 1), (skip, 0), (win_b, 1),
                              (tail, 0)])
            mw = pattern([(settle, 0), (region_a, 1), (region_b, 0)])
        else:
            pixel_next = pattern([(T - pn, 0), (pn, 1)])
            detect = pattern([(settle + skip, 0), (win_a, 1), (skip, 0), (win_b, 1),
                              (pn, 0)])
            mw = pattern([(settle, 0), (region_a, 1), (region_b + pn, 0)])
        # Laser HIGH if configured for scans OR a manual laser_on() is active (continuity,
        # answer-check A: a deliberately-lit laser must not drop LOW for the frame).
        laser_high = self._enable_laser or self._laser_is_on
        laser = pattern([(T, 1 if laser_high else 0)]) or [(T, 0)]

        # settle-aware per-window integration times for the c/s normalization (SCAN-007/008):
        # diff normalizes by window A exactly; sum by the ACTUAL total count-window duration
        # (window B is shorter by the tail — SCAN-008 LOW finding).
        self._win_time = win_a * 1e-9
        self._sum_time = (win_a + win_b) * 1e-9

        seq = self._pulser.createSequence()
        seq.setDigital(int(self._ps_channels['pixel_next']), pixel_next)
        seq.setDigital(int(self._ps_channels['detect']), detect)
        seq.setDigital(int(self._ps_channels['mw']), mw)
        seq.setDigital(int(self._ps_channels['laser']), laser)
        return seq

    def _available_pixels_total(self):
        """ Pixels physically complete so far. Authority = the CLOSED-window count from
        CountBetweenMarkers.getBinWidths() (>0 = window closed, independent of photon count —
        SCAN-002; nonzero-count proxies undercount with the laser off). Only in the LEGACY
        ordering may we snap to frame_size on NI AO is_task_done(): with pixel_next_at_start
        the last AO sample is clocked at the START of the last pixel, so the AO task finishes
        BEFORE that pixel's windows close (SCAN-008 HIGH). """
        if self._timed_out:
            return self._frame_size
        if self._cbm is None:
            return self._consumed
        closed_windows = int(np.count_nonzero(self._cbm.getBinWidths()))
        completed = closed_windows // 2
        if not self._pixel_next_at_start and self._ao_task is not None:
            try:
                if self._ao_task.is_task_done():
                    completed = self._frame_size
            except Exception:
                pass
        return min(completed, self._frame_size)

    def _pixels_from_windows(self, first_pixel, n_pixels):
        """ Convert CountBetweenMarkers windows to per-pixel c/s values.
        Window order: pixel0-A, pixel0-B, pixel1-A, ... (sequential detect edge pairs).
        sum  = (A+B) / (t_A + t_B)   (SCAN-008: normalize by the actual total window time)
        diff = (A-B) / t_A           (needs RF on; normalization known-open, design §8) """
        if n_pixels <= 0:
            return np.empty(0, dtype=np.float64)
        counts = np.asarray(self._cbm.getData(), dtype=np.float64)
        a = counts[0::2][first_pixel:first_pixel + n_pixels]
        b = counts[1::2][first_pixel:first_pixel + n_pixels]
        if self._scan_mode == 'diff':
            return (a - b) / self._win_time
        return (a + b) / self._sum_time

    def _drain_unread(self, number_of_samples):
        """ Post-stop drain (SCAN-003 stock contract): keep returning buffered/empty samples
        after the frame stopped; ValueError only when an explicit request exceeds what is
        left. Called with the thread lock held. """
        name = self._input_channel_name
        buffered = np.empty(0, dtype=np.float64) if self._unread is None \
            else self._unread[name]
        if number_of_samples is None:
            self._unread = {name: np.empty(0, dtype=np.float64)}
            return {name: buffered}
        if number_of_samples > len(buffered):
            raise ValueError(f'Requested {number_of_samples} samples but only '
                             f'{len(buffered)} buffered after stop.')
        self._unread = {name: buffered[number_of_samples:]}
        return {name: buffered[:number_of_samples]}

    def _teardown(self):
        """ Stop everything, leave outputs safe. Snapshot the not-yet-consumed pixels into
        the drain buffer BEFORE releasing the CountBetweenMarkers (SCAN-003, mirrors the
        stock __unread_samples_buffer). Called with the thread lock held. """
        # (1) snapshot remaining pixels for post-stop draining
        try:
            available = self._available_pixels_total()
            n_rest = max(0, available - self._consumed)
            values = self._pixels_from_windows(self._consumed, n_rest)
        except Exception:
            self.log.exception('Could not snapshot unread samples on teardown:')
            values = np.empty(0, dtype=np.float64)
        self._unread = {self._input_channel_name: values}
        self._consumed = 0

        # (2) Pulse Streamer -> idle/safe state (all LOW; laser only if owned + idle_laser_on)
        try:
            if self._pulser is not None:
                self._pulser.constant(self._idle_output_state())
        except Exception:
            self.log.exception('Could not stop the Pulse Streamer sequence:')

        # (3) NI AO task
        if self._ao_task is not None:
            try:
                with warnings.catch_warnings():
                    # nidaqmx warns when a finite task is stopped before all samples were
                    # emitted — expected on any aborted frame (stock module does the same).
                    warnings.simplefilter('ignore')
                    self._ao_task.stop()
            except Exception:
                self.log.exception('Error stopping the NI AO task:')
            try:
                self._ao_task.close()
            except Exception:
                self.log.exception('Error closing the NI AO task:')
            self._ao_task = None

        # (4) Time Tagger measurement (after the snapshot!)
        if self._cbm is not None:
            try:
                self._cbm.stop()
            except Exception:
                pass
            self._cbm = None
        if self._detect_monitor is not None:
            try:
                self._detect_monitor.stop()
            except Exception:
                pass
            self._detect_monitor = None
        self._combiner = None

    def _log_timeout_diagnostic(self):
        """ In-scan diagnostic (SCAN-001/004, design §9): localize a stuck frame — closed
        windows, AO state AND the live detect-edge rate tell 'markers not reaching the TT'
        (rate ~0: wiring/PS side) apart from 'window not closing' (rate ~2x sample rate:
        readout/CBM side). This method must never raise. """
        try:
            closed = int(np.count_nonzero(self._cbm.getBinWidths())) if self._cbm else -1
        except Exception:
            closed = -1
        try:
            ao_done = self._ao_task.is_task_done() if self._ao_task is not None else None
        except Exception:
            ao_done = None
        try:
            # NOTE (SCAN-004): Countrate averages since creation — after a long stall the
            # average understates a healthy live rate; interpret ~0 vs ~2x, not exact values.
            rate = float(self._detect_monitor.getData()[0]) \
                if self._detect_monitor is not None else float('nan')
        except Exception:
            rate = float('nan')
        expected = 2.0 * self._sample_rate
        self.log.error(
            f'Frame timeout after {time.time() - self._frame_start:.1f} s: '
            f'{closed} of {2 * self._frame_size} count windows closed, '
            f'NI AO task done: {ao_done}, consumed {self._consumed}/{self._frame_size} px, '
            f'detect-edge rate {rate:.1f}/s (expected ~{expected:.1f}/s while streaming; '
            f'~0 = markers not reaching the TT, ~expected = windows not closing/readout). '
            f'Returning remaining pixels as zeros instead of hanging (SCAN-001 backstop). '
            f'If windows are missing at high pixel rates, probe qudi-closed first '
            f'(SCAN-004 note) before changing readout code.')
