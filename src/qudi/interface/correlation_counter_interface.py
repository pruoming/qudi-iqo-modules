# -*- coding: utf-8 -*-
"""
Interface for a two-channel photon correlation counter (histogram of channel B clicks relative to
channel A clicks), as used for g2 / Hanbury Brown-Twiss measurements.

Written for the confocal_nv bench, 2026-10-30 (RUN-CONFOCAL-NV-G2-MODULE-STEP1-20261030). Only the
counting device: excitation (laser) control is not part of it.

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

__all__ = ['CorrelationCounterInterface', 'CorrelationCounterConstraints']

from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from qudi.core.module import Base


@dataclass(frozen=True)
class CorrelationCounterConstraints:
    """ channels: physical input channels that can be correlated; bin width (ps) and number of bins as
    inclusive (min, max). The number of bins is odd, so that tau = 0 is the centre bin. """
    channels: Tuple[int, ...]
    bin_width_ps: Tuple[int, int]
    n_bins: Tuple[int, int]


class CorrelationCounterInterface(Base):
    """ Two-channel correlation histogram. The histogram index tau runs symmetrically around 0:
    tau > 0 means channel B clicked after channel A. """

    @property
    @abstractmethod
    def constraints(self) -> CorrelationCounterConstraints:
        pass

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """ True while the histogram accumulates. """
        pass

    @property
    @abstractmethod
    def settings(self) -> Dict[str, object]:
        """ The applied settings: channels (A, B), bin_width_ps, n_bins. """
        pass

    @abstractmethod
    def configure(self, channels: Tuple[int, int], bin_width_ps: int, n_bins: int) -> Dict[str, object]:
        """ Set up a new, empty and stopped histogram. Not allowed while running. Returns the applied
        settings. """
        pass

    @abstractmethod
    def start(self) -> None:
        """ Start or continue accumulating into the current histogram. """
        pass

    @abstractmethod
    def stop(self) -> None:
        """ Pause accumulating; the histogram is kept. """
        pass

    @abstractmethod
    def clear(self) -> None:
        """ Empty the histogram and the capture time; the running state is unchanged. """
        pass

    @abstractmethod
    def get_data(self) -> Dict[str, object]:
        """ tau_ps (array), counts (array, raw coincidences per bin), g_norm (array, normalised to 1
        for uncorrelated clicks at the measured rates), capture_s (accumulation time in s). """
        pass

    @abstractmethod
    def get_count_rates(self) -> np.ndarray:
        """ Mean count rates (c/s) of channels A and B since the previous call (or since configure). """
        pass
