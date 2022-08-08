from locomotion.envs.sensors import sensor

import numpy as np
import typing

_ARRAY = typing.Iterable[float] #pylint: disable=invalid-name
_FLOAT_OR_ARRAY = typing.Union[float, _ARRAY] #pylint: disable=invalid-name
_DATATYPE_LIST = typing.Iterable[typing.Any] #pylint: disable=invalid-name

NUM_MOTORS = 12
NUM_LEGS = 4
JOINT_MAX_VELOCITIES = [52.4, 28.6, 28.6] * NUM_LEGS
JOINT_MAX_FORCES = [20, 55, 55] * NUM_LEGS

OBSERVED_VELOCITY_LOW = [-100] * NUM_MOTORS
OBSERVED_VELOCITY_HIGH = [100] * NUM_MOTORS

OBSERVED_TORQUE_LOW = [-100, -100, -100] * NUM_LEGS
OBSERVED_TORQUE_HIGH = [100, 100, 100] * NUM_LEGS

OBSERVED_FOOT_POS_LOW = [-1, -1, -1] * NUM_LEGS
OBSERVED_FOOT_POS_HIGH = [1, 1, 1] * NUM_LEGS

OBSERVED_ACCELERATION_LOW = -100
OBSERVED_ACCELERATION_HIGH = 100

OBSERVED_VEL_SHIFT_LOW = -200
OBSERVED_VEL_SHIFT_HIGH = 200
#JOINT_MAX_TORQUE = 50

class MotorVelocitySensor(sensor.BoxSpaceSensor):
    """A Sensor that reads motor velocities"""    

    def __init__(self,
                num_motors: int = NUM_MOTORS,
                noisy_reading: bool = True, 
                name: typing.Text = "MotorVelocities", 
                lower_bound: _FLOAT_OR_ARRAY = OBSERVED_VELOCITY_LOW,
                upper_bound: _FLOAT_OR_ARRAY = OBSERVED_VELOCITY_HIGH, 
                dtype: typing.Type[typing.Any] = np.float64) -> None:

        self._num_motors = num_motors
        self._noisy_reading = noisy_reading

        super(MotorVelocitySensor, self).__init__(
            name=name, 
            shape=(num_motors,), 
            lower_bound=lower_bound, 
            upper_bound=upper_bound, 
            dtype=dtype)
    
    def _get_observation(self) -> _ARRAY:
        if self._noisy_reading:
            motor_velocities = self._robot.GetMotorVelocities()
        else:
            motor_velocities = self._robot.GetTrueMotorVelocities()

        return motor_velocities
    
class MotorTorqueSensor(sensor.BoxSpaceSensor):
    def __init__(self,
            num_motors: int = NUM_MOTORS,
            noisy_reading: bool = True, 
            name: typing.Text = "MotorTorques", 
            lower_bound: _FLOAT_OR_ARRAY = OBSERVED_TORQUE_LOW,
            upper_bound: _FLOAT_OR_ARRAY = OBSERVED_TORQUE_HIGH, 
            dtype: typing.Type[typing.Any] = np.float64) -> None:

        self._num_motors = num_motors
        self._noisy_reading = noisy_reading

        super(MotorTorqueSensor, self).__init__(
            name=name, 
            shape=(num_motors,), 
            lower_bound=lower_bound, 
            upper_bound=upper_bound, 
            dtype=dtype)
    
    def _get_observation(self) -> _ARRAY:
        if self._noisy_reading:
            motor_torques = self._robot.GetMotorTorques()
        else:
            motor_torques = self._robot.GetTrueMotorTorques()

        return motor_torques

class FootPositionSensor(sensor.BoxSpaceSensor):
    def __init__(self,
        num_feet: int = NUM_LEGS,
        name: typing.Text = "FootPosition", 
        lower_bound: _FLOAT_OR_ARRAY = OBSERVED_FOOT_POS_LOW,
        upper_bound: _FLOAT_OR_ARRAY = OBSERVED_FOOT_POS_HIGH, 
        dtype: typing.Type[typing.Any] = np.float64) -> None:

        self._num_feet = num_feet

        super(FootPositionSensor, self).__init__(
            name=name, 
            shape=(num_feet * 3,), 
            lower_bound=lower_bound, 
            upper_bound=upper_bound, 
            dtype=dtype)
    
    def _get_observation(self) -> _ARRAY:
        motor_torques = self._robot.GetFootPositionsInBaseFrame().ravel()
        return motor_torques

class BaseAccelerationSensor(sensor.BoxSpaceSensor):
    """A Sensor that reads vase velocities and returns base acceleration"""    

    def __init__(self,
                num_motors: int = NUM_MOTORS,
                name: typing.Text = "BaseAcceleration", 
                lower_bound: _FLOAT_OR_ARRAY = OBSERVED_ACCELERATION_LOW,
                upper_bound: _FLOAT_OR_ARRAY = OBSERVED_ACCELERATION_HIGH, 
                dtype: typing.Type[typing.Any] = np.float64) -> None:

        
        self._channels = ["x", "y", "z"]
        self._num_channels = len(self._channels)

        super(BaseAccelerationSensor, self).__init__(
            name=name, 
            shape=(self._num_channels,), 
            lower_bound=lower_bound, 
            upper_bound=upper_bound, 
            dtype=dtype)
        
        datatype = [("{}_{}".format(name, channel), self._dtype)
                for channel in self._channels]
        self._datatype = datatype

        self._num_motors = num_motors
        self._last_velocity = np.zeros(self._num_channels)
        self._current_velocity = np.zeros(self._num_channels)

        
        
    def get_channels(self) -> typing.Iterable[typing.Text]:
        """Returns channels (displacement in x, y, z direction)."""
        return self._channels

    def get_num_channels(self) -> int:
        """Returns number of channels."""
        return self._num_channels

    def get_observation_datatype(self) -> _DATATYPE_LIST:
        """See base class."""
        return self._datatype
        
    def _get_observation(self) -> _ARRAY:

        accel = (self._current_velocity - self._last_velocity) / self._robot.time_step

        return accel
    
    def on_reset(self, env):
        self._last_velocity = np.zeros(self._num_channels)
        self._current_velocity = np.zeros(self._num_channels)

    def on_step(self, env):
        self._last_velocity = self._current_velocity
        self._current_velocity = np.array(self._robot.GetBaseVelocity())

class BaseVelocityShiftSensor(sensor.BoxSpaceSensor):
    """A Sensor that reads vase velocities and returns base acceleration"""    

    def __init__(self,
                num_motors: int = NUM_MOTORS,
                name: typing.Text = "BaseAcceleration", 
                lower_bound: _FLOAT_OR_ARRAY = OBSERVED_VEL_SHIFT_LOW,
                upper_bound: _FLOAT_OR_ARRAY = OBSERVED_VEL_SHIFT_HIGH, 
                dtype: typing.Type[typing.Any] = np.float64) -> None:

        
        self._channels = ["x", "y", "z"]
        self._num_channels = len(self._channels)

        super(BaseVelocityShiftSensor, self).__init__(
            name=name, 
            shape=(self._num_channels,), 
            lower_bound=lower_bound, 
            upper_bound=upper_bound, 
            dtype=dtype)
        
        datatype = [("{}_{}".format(name, channel), self._dtype)
                for channel in self._channels]
        self._datatype = datatype

        self._num_motors = num_motors
        self._last_velocity = np.zeros(self._num_channels)
        self._current_velocity = np.zeros(self._num_channels)

        
        
    def get_channels(self) -> typing.Iterable[typing.Text]:
        """Returns channels (displacement in x, y, z direction)."""
        return self._channels

    def get_num_channels(self) -> int:
        """Returns number of channels."""
        return self._num_channels

    def get_observation_datatype(self) -> _DATATYPE_LIST:
        """See base class."""
        return self._datatype
        
    def _get_observation(self) -> _ARRAY:

        #accel = (self._current_velocity - self._last_velocity) / self._robot.time_step
        shift = (self._current_velocity - self._last_velocity)
        return shift
    
    def on_reset(self, env):
        self._last_velocity = np.zeros(self._num_channels)
        self._current_velocity = np.zeros(self._num_channels)

    def on_step(self, env):
        self._last_velocity = self._current_velocity
        self._current_velocity = np.array(self._robot.GetBaseVelocity())