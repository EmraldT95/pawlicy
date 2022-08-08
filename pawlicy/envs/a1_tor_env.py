"""Wrapper to make the a1 environment suitable for OpenAI gym."""
import gym
import numpy as np
from locomotion.robots import a1, robot_config
from locomotion.envs import locomotion_gym_config, locomotion_gym_env
from locomotion.envs.sensors import robot_sensors
from locomotion.envs.env_wrappers import observation_dictionary_to_array_wrapper as obs_dict_to_array_wrapper, trajectory_generator_wrapper_env, simple_openloop

from pawlicy.robots import A1_loco
from pawlicy.sensors import a1_sensors

class A1TorEnv(gym.Env):
  """A1 Torque environment that supports the locomotion-gym interface."""
  metadata = {'render.modes': ['rgb_array']}

  def __init__(self,
                args,
                enable_rendering,
                task,
                wrap_trajectory_generator=True,
                action_limit=(0.50, 0.50, 0.50)):

    sim_params = locomotion_gym_config.SimulationParameters()
    sim_params.enable_rendering = enable_rendering
    # This is mainly for the action space and will be converted to torque by the motor model
    sim_params.motor_control_mode = robot_config.MotorControlMode.POSITION
    sim_params.reset_time = 2
    sim_params.num_action_repeat = 10
    sim_params.enable_action_interpolation = True
    sim_params.enable_action_filter = True
    sim_params.enable_clip_motor_commands = True
    sim_params.robot_on_rack = False
    #sim_params.randomise_terrain = args.randomise_terrain

    gym_config = locomotion_gym_config.LocomotionGymConfig(
      simulation_parameters=sim_params)

    sensors = [
        robot_sensors.BaseDisplacementSensor(dtype=np.float32),
        robot_sensors.IMUSensor(dtype=np.float32, channels=['R', 'P', 'Y', 'dR', 'dP', 'dY']),
        robot_sensors.MotorAngleSensor(num_motors=a1.NUM_MOTORS, dtype=np.float32),
        a1_sensors.MotorVelocitySensor(dtype=np.float32),
        a1_sensors.MotorTorqueSensor(dtype=np.float32)
    ]

    self._env = locomotion_gym_env.LocomotionGymEnv(gym_config=gym_config,
                                            robot_class=A1_loco,
                                            robot_sensors=sensors,
                                            task=task)

    self._env = obs_dict_to_array_wrapper.ObservationDictionaryToArrayWrapper(self._env)
    
    if wrap_trajectory_generator:
        self._env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(
            self._env,
            trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(
                action_limit=action_limit))

    self.observation_space = self._env.observation_space
    self.action_space = self._env.action_space

    self.args = args
    self.enable_rendering = enable_rendering

  def step(self, action):
    return self._env.step(action)

  def reset(self):
    return self._env.reset()

  def close(self):
    self._env.close()

  def render(self, mode):
    return self._env.render(mode)

  def __getattr__(self, attr):
    return getattr(self._env, attr)
