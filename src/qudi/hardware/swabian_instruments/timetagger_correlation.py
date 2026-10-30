# -*- coding: utf-8 -*-
"""
Two-channel photon correlation (g2 / HBT) on the shared Swabian Time Tagger.

Written for the confocal_nv bench, 2026-10-30 (RUN-CONFOCAL-NV-G2-MODULE-STEP1-20261030). Borrows the
Time Tagger from timetagger_provider (one connection for everything) and runs a TimeTagger Correlation
plus a Countrate on the two detector channels. The Time Tagger allows these next to the other
measurements on the same channels (time series, confocal and ODMR counting, pulsed fast counter).
This module never resets the tagger and never changes input delays or trigger levels: those are shared
settings. A cable delay between the detectors shows up as a shifted dip and is fitted.

Example config for copy-paste:

    timetagger_correlation:
        module.Class: 'swabian_instruments.timetagger_correlation.TimeTaggerCorrelation'
        connect:
            tagger_provider: timetagger_provider
        options:
            channels: [1, 2]          # detector A, detector B (APD1 -> ch1, APD2 -> ch2)
            bin_width_ps: 500
            n_bins: 8001              # odd; 8001 x 0.5 ns = +-2 us

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
import TimeTagger as tt

from qudi.core.configoption import ConfigOption
from qudi.core.connector import Connector
from qudi.util.mutex import Mutex
from qudi.interface.correlation_counter_interface import (CorrelationCounterInterface,
                                                          CorrelationCounterConstraints)


class TimeTaggerCorrelation(CorrelationCounterInterface):
    """ Correlation histogram of two Time Tagger input channels on the provider's shared tagger. """

    _tagger_provider = Connector(name='tagger_provider', interface='TimeTaggerProvider')

    _channels = ConfigOption(name='channels', default=[1, 2])
    _bin_width_ps = ConfigOption(name='bin_width_ps', default=500)
    _n_bins = ConfigOption(name='n_bins', default=8001)
    _input_channels = ConfigOption(name='input_channels', default=[1, 2, 3, 4, 5, 6, 7, 8])
    _max_n_bins = ConfigOption(name='max_n_bins', default=100001)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = Mutex()
        self._tagger = None
        self._corr = None
        self._rate = None
        self._settings = dict()
        self._constraints = None

    def on_activate(self):
        self._tagger = self._tagger_provider().get_tagger()
        self._constraints = CorrelationCounterConstraints(
            channels=tuple(int(c) for c in self._input_channels),
            bin_width_ps=(1, 10 ** 9),
            n_bins=(3, int(self._max_n_bins)))
        self.configure(tuple(self._channels), int(self._bin_width_ps), int(self._n_bins))
        self.log.info(f'Correlation on the shared Time Tagger, serial {self._tagger.getSerial()}, '
                      f'channels {self._settings["channels"]}.')

    def on_deactivate(self):
        with self._lock:
            self._release()
        self._tagger = None     # borrowed: the provider frees it

    @property
    def constraints(self):
        return self._constraints

    @property
    def is_running(self):
        with self._lock:
            return bool(self._corr is not None and self._corr.isRunning())

    @property
    def settings(self):
        with self._lock:
            return dict(self._settings)

    def configure(self, channels, bin_width_ps, n_bins):
        channels = tuple(int(c) for c in channels)
        bin_width_ps, n_bins = int(bin_width_ps), int(n_bins)
        if len(channels) != 2 or channels[0] == channels[1]:
            raise ValueError(f'two different channels needed, got {channels}')
        if not set(channels) <= set(self._constraints.channels):
            raise ValueError(f'channels {channels} not in {self._constraints.channels}')
        lo, hi = self._constraints.bin_width_ps
        if not lo <= bin_width_ps <= hi:
            raise ValueError(f'bin width {bin_width_ps} ps outside [{lo}, {hi}]')
        lo, hi = self._constraints.n_bins
        if not lo <= n_bins <= hi or n_bins % 2 == 0:
            raise ValueError(f'number of bins {n_bins} must be odd and within [{lo}, {hi}]')
        with self._lock:
            if self._corr is not None and self._corr.isRunning():
                raise RuntimeError('correlation is running; stop it before configure')
            self._release()
            self._corr = tt.Correlation(self._tagger, channels[0], channels[1], bin_width_ps, n_bins)
            self._corr.stop()           # a TimeTagger measurement starts on creation
            self._corr.clear()
            self._rate = tt.Countrate(self._tagger, list(channels))
            self._settings = dict(channels=channels, bin_width_ps=bin_width_ps, n_bins=n_bins)
        return self.settings

    def start(self):
        with self._lock:
            self._require_configured()
            self._corr.start()

    def stop(self):
        with self._lock:
            if self._corr is not None:
                self._corr.stop()

    def clear(self):
        with self._lock:
            self._require_configured()
            self._corr.clear()

    def get_data(self):
        with self._lock:
            self._require_configured()
            tau = np.asarray(self._corr.getIndex(), dtype=np.int64)
            counts = np.asarray(self._corr.getData(), dtype=np.int64)
            g_norm = np.asarray(self._corr.getDataNormalized(), dtype=float)
            capture_s = float(self._corr.getCaptureDuration()) / 1e12
        return dict(tau_ps=tau, counts=counts, g_norm=np.nan_to_num(g_norm), capture_s=capture_s)

    def get_count_rates(self):
        with self._lock:
            self._require_configured()
            rates = np.asarray(self._rate.getData(), dtype=float)
            self._rate.clear()
        return rates

    def _require_configured(self):
        if self._corr is None or self._rate is None:
            raise RuntimeError('no correlation set up (configure failed or module inactive)')

    def _release(self):
        for m in (self._corr, self._rate):
            if m is not None:
                try:
                    m.stop()
                except Exception:
                    pass
        self._corr = None
        self._rate = None
