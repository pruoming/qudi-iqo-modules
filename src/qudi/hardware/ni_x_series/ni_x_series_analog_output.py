# -*- coding: utf-8 -*-

"""
This file contains the qudi hardware module to use a National Instruments X-series card as mixed
signal input data streamer.

Qudi is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Qudi is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Qudi. If not, see <http://www.gnu.org/licenses/>.

Copyright (c) the Qudi Developers. See the COPYRIGHT.txt file at the
top-level directory of this distribution and at <https://github.com/Ulm-IQO/qudi/>
"""

# ToDo: Handle case where zero volts is not a good default value

import nidaqmx as ni

from qudi.util.mutex import Mutex
from qudi.core.configoption import ConfigOption
from qudi.core.statusvariable import StatusVar
from qudi.util.helpers import natural_sort, in_range

from qudi.interface.process_control_interface import ProcessControlConstraints
from qudi.interface.process_control_interface import ProcessSetpointInterface
from qudi.interface.mixins.process_control_switch import ProcessControlSwitchMixin
from qudi.hardware.ni_x_series.helpers import sanitize_device_name, normalize_channel_name
from qudi.hardware.ni_x_series.helpers import ao_channel_names, ao_voltage_range


# lele_ODMR_conf 2026-09-10 (review TASK-029, finding F2).
# A plain bool() is the wrong tool for a safety flag read out of a YAML config: bool('false')
# and bool('no') are both True, so a well-meant but quoted 'false' would SILENTLY re-enable an
# un-ramped voltage write to a piezo stage. That is the wrong direction to fail in. Refuse
# anything ambiguous instead, loudly, at config load time.
_TRUE_STRINGS = frozenset({'true', 'yes', 'on', '1'})
_FALSE_STRINGS = frozenset({'false', 'no', 'off', '0'})


def _strict_bool(value, option_name: str) -> bool:
    """ Interpret a config value as a bool, refusing anything ambiguous.

    Accepts real bools, the ints 0/1, and the case-insensitive strings
    true/false/yes/no/on/off/1/0. Raises ValueError on everything else.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in _TRUE_STRINGS:
            return True
        if norm in _FALSE_STRINGS:
            return False
    raise ValueError(
        f'Config option "{option_name}" must be an unambiguous boolean, got {value!r}. '
        f'Accepted: true/false, yes/no, on/off, 1/0 (any case), or a real YAML bool. '
        f'Refusing to guess: this option gates a voltage write to hardware, and guessing '
        f'wrong in the permissive direction would drive an actuator without being asked.'
    )


class NIXSeriesAnalogOutput(ProcessControlSwitchMixin, ProcessSetpointInterface):
    """ A module to output and read back (internally routed) analog voltages
    on an Ni Card in a software timed fashion.
    (only tested with Ni X-Series cards so far)

    Example config for copy-paste:

    nicard_63XX_ao:
        module.Class: 'ni_x_series.ni_x_series_analog_output.NIXSeriesAnalogOutput'
        options:
            device_name: 'Dev1'
            channels:
                ao0:
                    limits: [-10.0, 10.0]
                    keep_value: True
                ao1:
                    limits: [-10.0, 10.0]
                    keep_value: True
                ao2:
                    limits: [-10.0, 10.0]
                    keep_value: True
                ao3:
                    limits: [-10.0, 10.0]
                    keep_value: True
    """
    _device_name = ConfigOption(name='device_name',
                                default='Dev1',
                                missing='warn',
                                constructor=sanitize_device_name)
    _channels_config = ConfigOption(
        name='channels',
        default={
            'ao0': {'limits': (-10.0, 10.0), 'keep_value': True},
            'ao1': {'limits': (-10.0, 10.0), 'keep_value': True},
            'ao2': {'limits': (-10.0, 10.0), 'keep_value': True},
            'ao3': {'limits': (-10.0, 10.0), 'keep_value': True}
        },
        missing='warn'
    )

    # lele_ODMR_conf 2026-09-10 (TASK-027, revised after review TASK-029) -- SAFETY.
    #
    # Upstream writes self._setpoints to the hardware the moment a channel is activated
    # (see set_activity_state below). For a piezo stage that is a single un-ramped write to
    # whatever value happens to be persisted in the StatusVar, which has no guaranteed
    # relation to where the stage physically is: a reboot leaves the card at 0 V while the
    # StatusVar still claims the old position, and a missing StatusVar makes
    # _sanitize_setpoint_status fill 0 -- which, for limits [0, 10], is one END of travel.
    # The module has no readback (its own docstring claims one; there is no implementation),
    # so the mismatch cannot be detected. Upstream knew: see the ToDo at the top of this file.
    # With this option False, activating a channel creates the NI task but writes NOTHING,
    # so the output keeps whatever it already had.
    #
    # KNOW THE TRADE-OFF BEFORE SETTING THIS FALSE (review TASK-029 finding F1):
    # upstream's order is _create_ao_task and THEN _write_ao_value, so that write also
    # incidentally RESTORED whatever the card may have done to the output while the task was
    # being created, leaving hardware == belief. Setting this False removes that accidental
    # restore too. If the DAQmx idle default on your device is ZERO_VOLTS, the output is
    # parked at 0 V on every activation while _setpoints still claims the old value -- i.e.
    # the belief is wrong on EVERY startup, not only after a reboot, and no amount of reading
    # the persisted StatusVar can detect it because that IS the wrong belief. Measure the
    # actual output voltage on a scope or DMM before trusting either setting. See M-011.
    #
    # Default is True == upstream behaviour, deliberately: this fork stays drop-in for
    # configs that never heard of this option (e.g. software_pid_controller activates a
    # channel and then reads get_setpoint as though it had been applied). Setups that want
    # the safe behaviour say so explicitly, as lele_ODMR_conf's nicard_ao does.
    _write_setpoint_on_activation = ConfigOption(name='write_setpoint_on_activation',
                                                default=True,
                                                missing='nothing',
                                                constructor=lambda self, x: _strict_bool(
                                                    x, 'write_setpoint_on_activation'))

    _setpoints = StatusVar(name='current_setpoints', default=dict())

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._thread_lock = Mutex()

        self._constraints = None
        self._device_channel_mapping = dict()
        self._ao_task_handles = dict()
        self._keep_values = dict()
        # lele 2026-09-10 (review TASK-029, finding F3): channels already warned about, so a
        # permanent condition is reported once instead of on every activation. Activation is NOT
        # once per startup -- ni_scanning_probe_interfuse._prepare_movement activates and
        # _abort_cursor_movement deactivates on every cursor-move cycle, so an unconditional
        # warning here becomes several lines per mouse drag. Warnings nobody can act on are
        # warnings everybody learns to scroll past.
        self._idle_behaviour_warned = set()
        self._no_write_notified = set()

    def on_activate(self):
        """ Starts up the NI-card and performs sanity checks """
        # Check if device is connected and set device to use
        self._device_channel_mapping = dict()
        self._ao_task_handles = dict()
        self._keep_values = dict()
        self._idle_behaviour_warned = set()
        self._no_write_notified = set()

        # Sanitize channel configuration
        ao_limits = ao_voltage_range(self._device_name)
        valid_channels = ao_channel_names(self._device_name)
        valid_channels_lower = [name.lower() for name in valid_channels]
        limits = dict()
        for ch_name in natural_sort(self._channels_config):
            ch_cfg = self._channels_config[ch_name]
            norm_name = normalize_channel_name(ch_name).lower()
            try:
                device_name = valid_channels[valid_channels_lower.index(norm_name)]
            except (ValueError, IndexError):
                self.log.error(f'Invalid analog output channel "{ch_name}" configured. Channel '
                               f'will be ignored.\nValid analog output channels are: '
                               f'{valid_channels}')
                continue
            try:
                ch_limits = ch_cfg['limits']
            except KeyError:
                ch_limits = ao_limits
            else:
                if not all(in_range(lim, *ao_limits)[0] for lim in ch_limits):
                    self.log.error(
                        f'Invalid analog output voltage limits {ch_limits} configured for channel '
                        f'"{ch_name}". Channel will be ignored.\nValid analog output limits must '
                        f'lie in range {ao_limits}'
                    )
                    continue
            self._device_channel_mapping[ch_name] = device_name
            self._keep_values[ch_name] = bool(ch_cfg.get('keep_value', True))
            limits[ch_name] = ch_limits

        # Initialization of hardware constraints defined in the config file
        self._constraints = ProcessControlConstraints(
            setpoint_channels=self._device_channel_mapping,
            units={ch: 'V' for ch in self._device_channel_mapping},
            limits=limits,
            dtypes={ch: float for ch in self._device_channel_mapping}
        )

        # Sanitize status variables
        self._sanitize_setpoint_status()

    def on_deactivate(self):
        for channel in list(self._ao_task_handles):
            try:
                self._terminate_ao_task(channel)
            except:
                self.log.exception(f'Error while terminating NI analog out task "{channel}":')

    @property
    def constraints(self) -> ProcessControlConstraints:
        """ Read-Only property holding the constraints for this hardware module.
        See class ProcessControlConstraints for more details.
        """
        return self._constraints

    def set_activity_state(self, channel: str, active: bool) -> None:
        """ Set activity state for given channel.
        State is bool type and refers to active (True) and inactive (False).
        """
        try:
            active = bool(active)
        except Exception as err:
            raise TypeError('Unable to convert activity state to bool') from err
        with self._thread_lock:
            try:
                current_state = self._get_activity_state(channel)
            except KeyError as err:
                raise ValueError(f'Invalid channel specifier "{channel}". Valid channels are:\n'
                                 f'{self.constraints.all_channels}') from err
            if active != current_state:
                try:
                    if active:
                        self._create_ao_task(channel)
                        # lele 2026-09-10 (TASK-027): do NOT write on activation unless the
                        # config explicitly asks for it. See the ConfigOption above and M-010.
                        if self._write_setpoint_on_activation:
                            self._write_ao_value(channel, self._setpoints.get(channel, 0))
                        else:
                            # F3: say this once per channel, then drop to debug. Channels are
                            # re-activated on every cursor-move cycle.
                            msg = (f'AO channel "{channel}" activated WITHOUT writing a setpoint '
                                   f'(write_setpoint_on_activation=False). Hardware keeps its '
                                   f'existing output. Believed setpoint is '
                                   f'{self._setpoints.get(channel, 0)} V -- this is a BELIEF taken '
                                   f'from the StatusVar, NOT a measurement. It is wrong whenever '
                                   f'anything moved the output behind our back: a PC or card '
                                   f'reset, or the card parking the output at its DAQmx idle '
                                   f'default while the task was created. Reading the persisted '
                                   f'StatusVar cannot detect that -- measure the pin.')
                            if channel in self._no_write_notified:
                                self.log.debug(msg)
                            else:
                                self._no_write_notified.add(channel)
                                self.log.info(msg)
                    else:
                        self._terminate_ao_task(channel)
                finally:
                    self._update_module_state()

    def get_activity_state(self, channel: str) -> bool:
        """ Get activity state for given channel.
        State is bool type and refers to active (True) and inactive (False).
        """
        with self._thread_lock:
            return self._get_activity_state(channel)

    def _get_activity_state(self, channel: str) -> bool:
        if channel not in self.constraints.all_channels:
            raise ValueError(f'Invalid channel specifier "{channel}". Valid channels are:\n'
                             f'{self.constraints.all_channels}')
        return channel in self._ao_task_handles

    def _update_module_state(self) -> None:
        busy = len(self._ao_task_handles) > 0
        if busy and self.module_state() != 'locked':
            self.module_state.lock()
        elif not busy and self.module_state() == 'locked':
            self.module_state.unlock()

    def set_setpoint(self, channel: str, value: float) -> None:
        """ Set new setpoint for a single channel """
        value = float(value)
        with self._thread_lock:
            if not self._get_activity_state(channel):
                raise RuntimeError(f'Please activate channel "{channel}" before setting setpoint')
            if not self.constraints.channel_value_in_range(channel, value)[0]:
                raise ValueError(f'Setpoint {value} for channel "{channel}" out of allowed '
                                 f'value bounds {self.constraints.channel_limits[channel]}')
            self._write_ao_value(channel, value)
            self._setpoints[channel] = value

    def get_setpoint(self, channel: str) -> float:
        """ Get current setpoint for a single channel """
        with self._thread_lock:
            if not self._get_activity_state(channel):
                raise RuntimeError(f'Please activate channel "{channel}" before getting setpoint')
            return self._setpoints[channel]

    def _terminate_ao_task(self, channel: str) -> None:
        """ Reset analog output to 0 if keep_values flag is not set """
        try:
            if not self._keep_values[channel]:
                # lele 2026-09-10 (TASK-027) BUGFIX: upstream called self._write_ao_value(0),
                # passing a single positional arg to a (channel, value) signature. That raises
                # TypeError, which the KeyError-only except below does not catch, so the task was
                # neither popped nor closed: deactivate broke and leaked the NI task. Net effect
                # upstream: keep_value=False was unusable, i.e. "zero the output on shutdown"
                # silently was not an available mitigation at all. See mistakes/M-010.
                self._write_ao_value(channel, 0)
            task = self._ao_task_handles.pop(channel)
        except KeyError:
            return
        try:
            if not task.is_task_done():
                task.stop()
        finally:
            task.close()

    def _create_ao_task(self, channel: str) -> None:
        if channel in self._ao_task_handles:
            raise ValueError(f'AO task with name "{channel}" already present.')
        try:
            ao_task = ni.Task(channel)
        except ni.DaqError as err:
            raise RuntimeError(f'Unable to create NI task "{channel}"') from err
        try:
            ao_phys_ch = f'/{self._device_name}/{self._device_channel_mapping[channel]}'
            min_val, max_val = self.constraints.channel_limits[channel]
            ao_task.ao_channels.add_ao_voltage_chan(physical_channel=ao_phys_ch,
                                                    min_val=min_val,
                                                    max_val=max_val)
        except Exception as err:
            try:
                ao_task.close()
            except ni.DaqError:
                pass
            raise RuntimeError('Error while configuring NI analog out task') from err

        # lele 2026-09-10 (TASK-027) SAFETY: tell the card to hold its existing output while the
        # task is created and after it is cleared, instead of parking it at 0 V. Upstream never
        # set this, so the DAQmx default applied -- and if that default is ZERO_VOLTS then merely
        # creating this task drops the piezo to one end of travel before anybody writes anything.
        # Guarded on purpose: not every device or task type accepts the property, and a device
        # that refuses it must not stop the module from activating. Never assume it took effect;
        # the achieved value is read back and logged so the record reflects reality, not intent.
        # F6: catch only what a DEVICE refusal or an API change can raise. A bare `except
        # Exception` here would report a code-side fault (say a future nidaqmx renaming the enum)
        # as "the card may drive this output to its default", sending the operator to the bench to
        # chase a hardware ghost. That is exactly the misdirection M-008 cost us a whole round to
        # see through: never let an error message name a cause the evidence does not support.
        wanted = ni.constants.AOIdleOutputBehavior.MAINTAIN_EXISTING_VALUE
        try:
            ao_task.ao_channels[0].ao_idle_output_behavior = wanted
            # F6: actually COMPARE. Reading the value back and only printing it still assumes the
            # set took effect, which was the one thing this block promised not to do.
            achieved = ao_task.ao_channels[0].ao_idle_output_behavior
        except (ni.DaqError, AttributeError) as err:
            self._warn_idle_behaviour_once(
                channel,
                f'Could not set/verify ao_idle_output_behavior=MAINTAIN_EXISTING_VALUE on AO '
                f'channel "{channel}": {err}'
            )
        else:
            if achieved == wanted:
                self.log.debug(f'AO channel "{channel}" idle output behaviour: {achieved}.')
            else:
                self._warn_idle_behaviour_once(
                    channel,
                    f'AO channel "{channel}" idle output behaviour did not take: asked for '
                    f'{wanted}, device reports {achieved}'
                )

        self._ao_task_handles[channel] = ao_task

    def _warn_idle_behaviour_once(self, channel: str, detail: str) -> None:
        """ Report an idle-output-behaviour problem once per channel, then keep it at debug.

        lele 2026-09-10 (review TASK-029, finding F3). The condition is usually permanent -- a
        device either supports the property or it never will -- while activation repeats on every
        cursor-move cycle. Warning every time trains the operator to ignore the log, which is the
        opposite of what a hardware warning is for.
        """
        message = (
            f'{detail}. This output may therefore be driven to the DAQmx idle default (possibly '
            f'0 V, i.e. one END of travel for a unipolar piezo range) when the task is created '
            f'or cleared, without anything asking it to. For a piezo axis or an AOM line that is '
            f'a real hardware hazard: measure the pin on a scope or DMM before moving or scanning '
            f'-- do not infer the output from the persisted setpoint. See mistakes/M-011.'
        )
        if channel in self._idle_behaviour_warned:
            self.log.debug(message)
        else:
            self._idle_behaviour_warned.add(channel)
            self.log.warning(message)

    def _write_ao_value(self, channel: str, value: float) -> None:
        self._ao_task_handles[channel].write(value)

    def _sanitize_setpoint_status(self) -> None:
        # Remove obsolete channels and out-of-bounds values
        for channel, value in list(self._setpoints.items()):
            try:
                if not self.constraints.channel_value_in_range(channel, value)[0]:
                    del self._setpoints[channel]
            except KeyError:
                del self._setpoints[channel]
        # Add missing setpoint channels and set initial value to zero
        self._setpoints.update(
            {ch: 0 for ch in self.constraints.setpoint_channels if ch not in self._setpoints}
        )
