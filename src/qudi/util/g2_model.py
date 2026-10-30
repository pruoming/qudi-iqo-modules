# -*- coding: utf-8 -*-
"""
g2 (Hanbury Brown-Twiss) model, fit and background correction for single emitters such as NV centres.

Moved from the bench script agent-pipeline/temp/g2_20261030/g2.py (tested on B3_-1_2, 2026-10-30) for the
g2 module (RUN-CONFOCAL-NV-G2-MODULE-STEP1-20261030). Used by g2_logic and correlation_counter_dummy.

Physics. With signal S and uncorrelated background B, rho = S / (S + B) and the measured
g2_raw(t) = 1 - rho^2 + rho^2 g2_NV(t); for N equal emitters g2_NV(0) = 1 - 1/N. So a single emitter gives
at best g2_raw(0) = 1 - rho^2. The background-corrected g2_c = (g2_raw - (1 - rho^2)) / rho^2.
The fit model is the three-level one, 1 - p[(1 + a) exp(-|t - t0|/t1) - a exp(-|t - t0|/t2)], times a free
level c (normalisation at long delay, bunching included), convolved with a Gaussian timing jitter sigma
(APD + tagger, an assumption); g2(0) = 1 - p. t0 is the cable delay between the two detectors.

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

__all__ = ['SIGMA_NS', 'DIP_HALF_NS', 'raw_floor', 'correct', 'three_level', 'model', 'fit_g2']

import numpy as np
from scipy.optimize import curve_fit

SIGMA_NS = 0.5          # timing jitter of the detector pair (two APDs ~0.35 ns each), assumed
DIP_HALF_NS = 0.75      # raw dip: bins within +-0.75 ns of the fitted centre


def raw_floor(rho, n=1):
    """ Best raw g2(0) for n equal emitters at signal fraction rho. """
    return 1.0 - rho ** 2 / n


def correct(g2, g2_err, rho, rho_err):
    """ Background-corrected g2 and its error (g2 error and rho error combined). """
    gc = (g2 - (1.0 - rho ** 2)) / rho ** 2
    d_g = 1.0 / rho ** 2
    d_rho = 2.0 * (1.0 - g2) / rho ** 3
    return float(gc), float(np.hypot(d_g * g2_err, d_rho * rho_err))


def three_level(t_ns, p, a, t1, t2, t0):
    x = np.abs(t_ns - t0)
    return 1.0 - p * ((1.0 + a) * np.exp(-x / t1) - a * np.exp(-x / t2))


def model(t_ns, c, p, a, t1, t2, t0, sigma=SIGMA_NS):
    """ c x three-level g2 convolved with a Gaussian of sigma (ns), on a 0.05 ns grid. """
    t_ns = np.asarray(t_ns, float)
    if sigma <= 0:
        return c * three_level(t_ns, p, a, t1, t2, t0)
    k = np.arange(-4 * sigma, 4 * sigma + 1e-9, 0.05)
    w = np.exp(-0.5 * (k / sigma) ** 2)
    w /= w.sum()
    return c * np.sum(w[None, :] * three_level(t_ns[:, None] + k[None, :], p, a, t1, t2, t0), axis=1)


def fit_g2(t_ns, g, counts, sigma=SIGMA_NS, dip_half_ns=DIP_HALF_NS):
    """ Fit the normalised histogram g (counts per bin give the Poisson errors). Returns a dict or None
    (fewer than 50 non-empty bins, or no start value converged).

    g2_0 / dip_bins are relative to the fitted long-delay level c; g2_0_tt / dip_bins_tt are relative to
    the counter's own normalisation (level 1 for uncorrelated clicks). The latter is the better figure when
    the bunching time t2 exceeds the histogram window, because c is then an extrapolation. """
    t_ns, g, counts = (np.asarray(v, float) for v in (t_ns, g, counts))
    ok = counts > 0
    if ok.sum() < 50:
        return None
    err = np.where(ok, g / np.sqrt(np.maximum(counts, 1)), np.inf)
    far = np.abs(t_ns) > 0.6 * np.abs(t_ns).max()
    c0 = float(np.median(g[far & ok])) if np.any(far & ok) else 1.0
    near = np.abs(t_ns) < 60
    sm = np.convolve(g, np.ones(9) / 9, mode='same')
    t00 = float(t_ns[near][np.argmin(sm[near])]) if near.any() else 0.0
    p0 = float(np.clip(1 - sm[near].min() / c0, 0.05, 0.95)) if near.any() else 0.3
    best = None
    for t1 in (5.0, 12.0, 25.0):
        for t2 in (80.0, 250.0):
            try:
                popt, pcov = curve_fit(lambda t, c, p, a, t1_, t2_, t0: model(t, c, p, a, t1_, t2_, t0, sigma),
                                       t_ns[ok], g[ok], p0=[c0, p0, 0.3, t1, t2, t00], sigma=err[ok],
                                       absolute_sigma=True,
                                       bounds=([0.2, 0.0, 0.0, 0.3, 20.0, t00 - 15],
                                               [5.0, 1.0, 5.0, 60.0, 3000.0, t00 + 15]),
                                       maxfev=20000)
            except (RuntimeError, ValueError):
                continue
            chi2 = float(np.sum(((g[ok] - model(t_ns[ok], *popt, sigma=sigma)) / err[ok]) ** 2))
            if best is None or chi2 < best[0]:
                best = (chi2, popt, pcov)
    if best is None:
        return None
    chi2, popt, pcov = best
    perr = np.sqrt(np.clip(np.diag(pcov), 0, None))
    c, p, a, t1, t2, t0 = (float(v) for v in popt)
    # raw dip straight from the bins around the fitted centre, normalised to the fitted long-delay level
    sel = ok & (np.abs(t_ns - t0) <= dip_half_ns)
    dip = float(np.sum(g[sel]) / sel.sum() / c) if sel.any() else float('nan')
    dip_err = float(dip / np.sqrt(np.sum(counts[sel]))) if sel.any() else float('nan')
    return dict(g2_0=1.0 - p, g2_0_err=float(perr[1]), level=c, a=a, t1_ns=t1, t1_err_ns=float(perr[3]),
                t2_ns=t2, t0_ns=t0, t0_err_ns=float(perr[5]), sigma_ns=sigma,
                dip_bins=dip, dip_bins_err=dip_err, dip_bins_n=int(sel.sum()),
                chi2_red=chi2 / max(1, int(ok.sum()) - 6), counts_at_dip=int(np.sum(counts[sel])),
                g2_0_tt=c * (1.0 - p), dip_bins_tt=dip * c, dip_bins_tt_err=dip_err * c)
