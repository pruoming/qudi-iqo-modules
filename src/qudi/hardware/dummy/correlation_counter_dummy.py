# -*- coding: utf-8 -*-
"""
Dummy two-channel correlation counter: a single emitter with background, seen through a 50:50 HBT
splitter. Synthetic data only; describes no bench.

Written for the g2 module, 2026-10-30 (RUN-CONFOCAL-NV-G2-MODULE-STEP1-20261030), for the desk config and
the tests. The histogram grows with the time the counter runs (Poisson counts drawn when the data are
read): expected coincidences per bin = rate_A * rate_B * bin width * time * g2_raw(tau), with
g2_raw = 1 - rho^2 + rho^2 g2_emitter(tau) and the three-level emitter model of qudi.util.g2_model.
set_excitation(False) mimics a lost laser: the rates fall to the dark rates and the histogram flattens.

Example config for copy-paste:

    correlation_counter_dummy:
        module.Class: 'dummy.correlation_counter_dummy.CorrelationCounterDummy'
        options:
            rates: [40.0e3, 46.0e3]   # c/s per detector with the laser on (signal + background)
            dark_rates: [100, 500]    # c/s per detector with the laser off
            rho: 0.69                 # signal fraction S / (S + B)
            emitters: 1               # number of equal emitters (g2_emitter(0) = 1 - 1/emitters)
            t1_ns: 7.0
            t2_ns: 800.0
            bunching: 0.4             # three-level parameter a
            delay_ns: 3.5             # cable delay B - A (dip position)

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

import time

import numpy as np

from qudi.core.configoption import ConfigOption
from qudi.util.mutex import Mutex
from qudi.util.g2_model import model, SIGMA_NS
from qudi.interface.correlation_counter_interface import (CorrelationCounterInterface,
                                                          CorrelationCounterConstraints)


class CorrelationSim:
    """ The simulation without the qudi module machinery (used by the dummy and by the tests). """

    def __init__(self, rates=(40.0e3, 46.0e3), dark_rates=(100.0, 500.0), rho=0.69, emitters=1,
                 t1_ns=7.0, t2_ns=800.0, bunching=0.4, delay_ns=3.5, sigma_ns=SIGMA_NS, seed=None,
                 clock=time.monotonic):
        self.rates = np.asarray(rates, float)
        self.dark_rates = np.asarray(dark_rates, float)
        self.rho = float(rho)
        self.p = 1.0 / max(1, int(emitters))
        self.shape_args = dict(a=float(bunching), t1=float(t1_ns), t2=float(t2_ns), t0=float(delay_ns),
                               sigma=float(sigma_ns))
        self.excitation = True
        self._rng = np.random.default_rng(seed)
        self._clock = clock
        self.settings = dict()
        self._g_raw = None
        self._counts = None
        self._capture_s = 0.0
        self._norm_rate2 = 0.0      # sum over segments of rate_A * rate_B * duration (normalisation)
        self._running_since = None
        self._rate_window = [np.zeros(2), 0.0]      # counts and time since the last rate read
        self._rate_since = self._clock()

    def configure(self, channels, bin_width_ps, n_bins):
        self._advance()
        self.settings = dict(channels=tuple(int(c) for c in channels), bin_width_ps=int(bin_width_ps),
                             n_bins=int(n_bins))
        tau_ns = self.tau_ps() / 1e3
        a = self.shape_args
        emitter = model(tau_ns, 1.0, self.p, a['a'], a['t1'], a['t2'], a['t0'], sigma=a['sigma'])
        self._g_raw = 1.0 - self.rho ** 2 + self.rho ** 2 * emitter
        self._counts = np.zeros(int(n_bins), dtype=np.int64)
        self._capture_s = 0.0
        self._norm_rate2 = 0.0
        self._running_since = None
        self._rate_window = [np.zeros(2), 0.0]
        return dict(self.settings)

    def tau_ps(self):
        n, bw = self.settings['n_bins'], self.settings['bin_width_ps']
        return (np.arange(n, dtype=np.int64) - n // 2) * bw

    def current_rates(self):
        return self.rates.copy() if self.excitation else self.dark_rates.copy()

    @property
    def is_running(self):
        return self._running_since is not None

    def start(self):
        if self._running_since is None:
            self._running_since = self._clock()

    def stop(self):
        self._advance()
        self._running_since = None

    def clear(self):
        self._advance()
        self._counts[:] = 0
        self._capture_s = 0.0
        self._norm_rate2 = 0.0

    def set_excitation(self, on):
        """ Laser on/off for the simulation; the time so far is accounted with the old state. """
        self._advance()
        self.excitation = bool(on)

    def get_data(self):
        self._advance()
        if self._norm_rate2 > 0:
            g_norm = self._counts / (self._norm_rate2 * self.settings['bin_width_ps'] * 1e-12)
        else:
            g_norm = np.zeros(self._counts.size)
        return dict(tau_ps=self.tau_ps(), counts=self._counts.copy(), g_norm=g_norm,
                    capture_s=float(self._capture_s))

    def get_count_rates(self):
        self._advance()
        counts, seconds = self._rate_window
        self._rate_window = [np.zeros(2), 0.0]
        return counts / seconds if seconds > 0 else self.current_rates()

    def _advance(self):
        """ Account for the time since the last call with the current state. """
        now = self._clock()
        dt_rate = now - self._rate_since
        self._rate_since = now
        if dt_rate > 0:
            r = self.current_rates()
            self._rate_window[0] = self._rate_window[0] + self._rng.poisson(r * dt_rate)
            self._rate_window[1] += dt_rate
        if self._running_since is None or self._counts is None:
            return
        dt = now - self._running_since
        self._running_since = now
        if dt <= 0:
            return
        r = self.current_rates()
        bw_s = self.settings['bin_width_ps'] * 1e-12
        if self.excitation:
            lam = r[0] * r[1] * bw_s * dt * self._g_raw
        else:
            lam = np.full(self._counts.size, r[0] * r[1] * bw_s * dt)     # dark counts: uncorrelated
        self._counts += self._rng.poisson(lam)
        self._capture_s += dt
        self._norm_rate2 += r[0] * r[1] * dt


class CorrelationCounterDummy(CorrelationCounterInterface):
    """ Synthetic correlation counter (see the module docstring). """

    _channels = ConfigOption(name='channels', default=[1, 2])
    _bin_width_ps = ConfigOption(name='bin_width_ps', default=500)
    _n_bins = ConfigOption(name='n_bins', default=8001)
    _rates = ConfigOption(name='rates', default=[40.0e3, 46.0e3])
    _dark_rates = ConfigOption(name='dark_rates', default=[100.0, 500.0])
    _rho = ConfigOption(name='rho', default=0.69)
    _emitters = ConfigOption(name='emitters', default=1)
    _t1_ns = ConfigOption(name='t1_ns', default=7.0)
    _t2_ns = ConfigOption(name='t2_ns', default=800.0)
    _bunching = ConfigOption(name='bunching', default=0.4)
    _delay_ns = ConfigOption(name='delay_ns', default=3.5)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = Mutex()
        self._sim = None
        self._constraints = None

    def on_activate(self):
        self._sim = CorrelationSim(rates=self._rates, dark_rates=self._dark_rates, rho=self._rho,
                                   emitters=self._emitters, t1_ns=self._t1_ns, t2_ns=self._t2_ns,
                                   bunching=self._bunching, delay_ns=self._delay_ns)
        self._constraints = CorrelationCounterConstraints(channels=(1, 2, 3, 4, 5, 6, 7, 8),
                                                          bin_width_ps=(1, 10 ** 6), n_bins=(3, 100001))
        self.configure(tuple(self._channels), int(self._bin_width_ps), int(self._n_bins))

    def on_deactivate(self):
        self._sim = None

    @property
    def constraints(self):
        return self._constraints

    @property
    def is_running(self):
        with self._lock:
            return self._sim is not None and self._sim.is_running

    @property
    def settings(self):
        with self._lock:
            return dict(self._sim.settings) if self._sim is not None else dict()

    def configure(self, channels, bin_width_ps, n_bins):
        channels = tuple(int(c) for c in channels)
        if len(channels) != 2 or channels[0] == channels[1]:
            raise ValueError(f'two different channels needed, got {channels}')
        if not set(channels) <= set(self._constraints.channels):
            raise ValueError(f'channels {channels} not in {self._constraints.channels}')
        lo, hi = self._constraints.bin_width_ps
        if not lo <= int(bin_width_ps) <= hi:
            raise ValueError(f'bin width {bin_width_ps} ps outside [{lo}, {hi}]')
        if int(n_bins) % 2 == 0 or not 3 <= int(n_bins) <= self._constraints.n_bins[1]:
            raise ValueError(f'number of bins {n_bins} must be odd and within {self._constraints.n_bins}')
        with self._lock:
            if self._sim.is_running:
                raise RuntimeError('correlation is running; stop it before configure')
            return self._sim.configure(channels, bin_width_ps, n_bins)

    def start(self):
        with self._lock:
            self._sim.start()

    def stop(self):
        with self._lock:
            self._sim.stop()

    def clear(self):
        with self._lock:
            self._sim.clear()

    def get_data(self):
        with self._lock:
            return self._sim.get_data()

    def get_count_rates(self):
        with self._lock:
            return self._sim.get_count_rates()

    def set_excitation(self, on):
        """ Dummy only: simulate the laser switching on or off. """
        with self._lock:
            self._sim.set_excitation(on)
