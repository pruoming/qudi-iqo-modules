# -*- coding: utf-8 -*-
"""
Derived fit results (2026-10-24), shown with the fitted parameters and saved with them.

Line-shape fits (Lorentzian / Gaussian; single, double, triple):

    linewidth (FWHM) = 2 * sigma               (Lorentzian; qudi's sigma is the half width)
                     = 2 * sqrt(2 ln 2) * sigma (Gaussian; sigma is the standard deviation)
    contrast         = |amplitude| / offset     (in %, relative to the baseline)

Rabi fits (single-frequency Sine, ExponentialDecaySine; frequency f = Rabi frequency):

    pi pulse = 1 / (2 f),  pi/2 pulse = 1 / (4 f),  3pi/2 pulse = 3 / (4 f),
    Rabi period (2pi) = 1 / f   (the 'rabi_period' the predefined pulsed methods take)

Errors are propagated from the fitted parameter errors without covariances. In double/triple
line fits each line's contrast is that component's depth relative to the common baseline (not
the visible depth where lines overlap). No contrast is given for a baseline <= 0 (e.g.
difference signals); the *Linear line models and double-frequency sines are not covered. The
pulse lengths follow from the frequency alone: a dead time or phase offset in the data shifts
the observed first minimum, not these values.

DerivedFitContainer is a FitContainer whose formatted_result / dict_result include these values,
so the stock fit widgets (which call container.formatted_result) show them unchanged.
"""

import math

import numpy as np

from qudi.util.datafitting import FitContainer
from qudi.util.units import create_formatted_output

__all__ = ['derived_line_parameters', 'derived_pulse_parameters', 'derived_parameters',
           'DerivedFitContainer']

_FWHM_FACTOR = {'Lorentzian': 2.0, 'Gaussian': 2.0 * math.sqrt(2.0 * math.log(2.0))}
_LINES = {'': 1, 'Double': 2, 'Triple': 3}
_RABI_MODELS = ('Sine', 'ExponentialDecaySine')
# (label, pulse length in units of the Rabi period)
_PULSES = (('pi pulse', 0.5), ('pi/2 pulse', 0.25), ('3pi/2 pulse', 0.75),
           ('Rabi period (2pi)', 1.0))


def _model_name(fit_result):
    return type(getattr(fit_result, 'model', None)).__name__


def _line_model(fit_result):
    """ (shape, number of lines) for a supported model, else None. """
    name = _model_name(fit_result)
    for prefix, n_lines in _LINES.items():
        for shape in _FWHM_FACTOR:
            if name == prefix + shape:
                return shape, n_lines
    return None


def _value_error(params, name):
    param = params[name]
    error = param.stderr if param.vary else 0.0
    return float(param.value), (np.nan if error is None else float(error))


def derived_line_parameters(fit_result, x_unit=''):
    """ {'linewidth (FWHM)[ n]': {value, error, unit}, 'contrast[ n]': {...}} for Lorentzian /
    Gaussian line fits (n = line number for double/triple models); {} for other models. """
    model = _line_model(fit_result) if fit_result is not None else None
    if model is None:
        return dict()
    shape, n_lines = model
    params = fit_result.params
    factor = _FWHM_FACTOR[shape]
    offset, offset_err = _value_error(params, 'offset')
    derived = dict()
    for line in range(1, n_lines + 1):
        suffix = '' if n_lines == 1 else f'_{line}'
        label = '' if n_lines == 1 else f' {line}'
        sigma, sigma_err = _value_error(params, 'sigma' + suffix)
        derived['linewidth (FWHM)' + label] = {'value': factor * sigma,
                                               'error': factor * sigma_err, 'unit': x_unit}
        if offset > 0:
            amp, amp_err = _value_error(params, 'amplitude' + suffix)
            contrast = abs(amp) / offset
            error = math.sqrt((amp_err / offset) ** 2 + (contrast * offset_err / offset) ** 2)
            derived['contrast' + label] = {'value': 100 * contrast, 'error': 100 * error,
                                           'unit': '%'}
    return derived


def derived_pulse_parameters(fit_result, x_unit=''):
    """ {'pi pulse': {value, error, unit}, 'pi/2 pulse', '3pi/2 pulse', 'Rabi period (2pi)'}
    from the frequency of a single-frequency Rabi fit (Sine, ExponentialDecaySine); {} else. """
    if fit_result is None or _model_name(fit_result) not in _RABI_MODELS:
        return dict()
    if x_unit not in ('s', ''):
        return dict()            # a sine on a frequency (or other) axis has no pulse lengths
    freq, freq_err = _value_error(fit_result.params, 'frequency')
    if not freq > 0:
        return dict()
    period, period_err = 1.0 / freq, freq_err / freq ** 2
    return {label: {'value': fraction * period, 'error': fraction * period_err, 'unit': x_unit}
            for label, fraction in _PULSES}


def derived_parameters(fit_result, x_unit=''):
    """ All derived values of a fit result (line shape and Rabi pulse lengths). """
    derived = derived_line_parameters(fit_result, x_unit)
    derived.update(derived_pulse_parameters(fit_result, x_unit))
    return derived


def _percent(value, error):
    """ '3.05 ± 0.08 %' with the error rounded to 2 significant digits. """
    if error is None or np.isnan(error):
        return f'{value:.3g} ± nan %'           # fitted, but no error estimate
    if not np.isfinite(error) or error <= 0:
        return f'{value:.3g} %'
    decimals = max(0, 1 - int(math.floor(math.log10(error))))
    return f'{value:.{decimals}f} ± {error:.{decimals}f} %'


class DerivedFitContainer(FitContainer):
    """ FitContainer that also reports linewidth (FWHM) and contrast of line-shape fits and the
    pi, pi/2, 3pi/2 pulse lengths and Rabi period of Rabi fits.

    @param x_unit: unit of the fitted x axis (str), or a callable returning it (e.g. the pulsed
                   measurement's current data units)
    """

    def __init__(self, *args, x_unit='', **kwargs):
        super().__init__(*args, **kwargs)
        self._x_unit = x_unit

    @property
    def x_unit(self):
        unit = self._x_unit() if callable(self._x_unit) else self._x_unit
        return unit or ''

    def formatted_result(self, fit_result, parameters_units=None):
        text = FitContainer.formatted_result(fit_result, parameters_units)
        derived = derived_parameters(fit_result, self.x_unit)
        if not derived:
            return text
        lines = [text.rstrip('\n')]
        with_unit = {k: v for k, v in derived.items() if v['unit'] != '%'}
        if with_unit:
            lines.append(create_formatted_output(with_unit).rstrip('\n'))
        # percent without SI prefix ('3.05 +- 0.08 %', not '3050 m%')
        lines += [f'{name}: {_percent(item["value"], item["error"])}'
                  for name, item in derived.items() if item['unit'] == '%']
        return '\n'.join(lines) + '\n'

    def dict_result(self, fit_result, parameters_units=None, export_keys=('value', 'stderr')):
        result = FitContainer.dict_result(fit_result, parameters_units, export_keys)
        for name, item in derived_parameters(fit_result, self.x_unit).items():
            # no '%' in saved results: qudi writes data-file metadata through configparser,
            # which rejects a bare '%' (invalid interpolation) and the whole save fails
            unit = 'percent' if item['unit'] == '%' else item['unit']
            result[name] = {'value': item['value'], 'stderr': item['error'], 'unit': unit}
        return result
