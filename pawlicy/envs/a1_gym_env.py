# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This file implements the locomotion gym env."""
import collections
import time
import pkgutil
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bullet_client
import pybullet_data as pd


from pawlicy.robots import robot_config
from pawlicy.robots.a1 import a1
from pawlicy.sensors import sensor
from pawlicy.sensors import space_utils

from pawlicy.envs.terrains import constants as terrain_constants
from pawlicy.envs.terrains.randomizer import TerrainRandomizer

_ACTION_EPS = 0.01
_NUM_SIMULATION_ITERATION_STEPS = 300
_LOG_BUFFER_LENGTH = 5000


class A1GymEnv(gym.Env):
	"""The gym environment for the locomotion tasks."""
	metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 66}

	def __init__(self,
				gym_config,
				robot_sensors=None,
				task=None):
		"""Initializes the locomotion gym environment.

		Args:
			gym_config: An instance of LocomotionGymConfig.
			sensors: A list of sensors for observation.
			task: A callable function/class to calculate the reward and termination
				condition. Takes the gym env as the argument when calling.

		Raises:
			ValueError: If the num_action_repeat is less than 1.

		"""

		self.seed(1)
		self._gym_config = gym_config
		self._world_dict = {} # A dictionary containing the objects in the world other than the robot.
		self._task = task	

		# Configure PyBullet
		self._configureSimulator(self._gym_config)

		# The action list contains the name of all actions.
		self._build_action_space()
		# The observation space consists of all the values from the sensors
		self._build_observation_space(robot_sensors)

		# Hard reset initially to load the robot URDF file
		self._hard_reset = True
		self.reset()
		self._hard_reset = gym_config.enable_hard_reset

	def reset(self,
				initial_motor_angles=None,
				reset_duration=0.0,
				reset_visualization_camera=True):
		"""Resets the robot's position in the world or rebuild the sim world.

		The simulation world will be rebuilt if self._hard_reset is True.

		Args:
			initial_motor_angles: A list of Floats. The desired joint angles after
			reset. If None, the robot will use its built-in value.
			reset_duration: Float. The time (in seconds) needed to rotate all motors
			to the desired initial values.
			reset_visualization_camera: Whether to reset debug visualization camera on
			reset.

		Returns:
			A numpy array contains the initial observation after reset.
		"""
		if self._is_render:
			self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, 0)

		# Clear the simulation world and rebuild the robot interface.
		if self._hard_reset:
			self._pybullet_client.resetSimulation()
			self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=self._num_bullet_solver_iterations)
			self._pybullet_client.setTimeStep(self._sim_time_step)
			self._pybullet_client.setGravity(0, 0, -10)

			# Randomising terrains
			terrain_id = -1
			if self._randomise_terrain:
				terrain_id, terrain_type = self._terrain_randomizer.randomize()
			else:
				terrain_id = self._pybullet_client.loadURDF("plane_implicit.urdf")
				terrain_type = "plane"

			# Rebuild the world.
			self._world_dict = { 
				"ground": {
					"id": terrain_id,
					"type": terrain_type
				}
			}

			# Rebuild the robot
			self._robot = a1.A1(
				pybullet_client=self._pybullet_client,
				enable_clip_motor_commands=self._gym_config.enable_clip_motor_commands,
				action_repeat=self._num_action_repeat,
				time_step=self._sim_time_step,
				sensors=self._robot_sensors,
				on_rack=self._on_rack,
				enable_action_interpolation=self._gym_config.enable_action_interpolation,
				enable_action_filter=self._gym_config.enable_action_filter,
				reset_time=self._gym_config.reset_time,
				allow_knee_contact=self._gym_config.allow_knee_contact,
				motor_control_mode=self._gym_config.motor_control_mode,
				init_position=terrain_constants.ROBOT_INIT_POSITION[terrain_type])
		else:
			# Reset the pose of the robot.
			self._robot.Reset(reload_urdf=False, default_motor_angles=initial_motor_angles, reset_time=reset_duration)

		self._pybullet_client.setPhysicsEngineParameter(enableConeFriction=0)
		self._env_step_counter = 0
		if reset_visualization_camera:
			self._pybullet_client.resetDebugVisualizerCamera(self._camera_dist, self._camera_yaw, self._camera_pitch, [0, 0, 0])
		self._last_action = np.zeros(self.action_space.shape)

		if self._is_render:
			self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, 1)

		for s in self.all_sensors():
			s.on_reset(self)

		if self._task and hasattr(self._task, 'reset'):
			self._task.reset(self)

		return self._get_observation()

	def step(self, action):
		"""Step forward the simulation, given the action.

		Args:
			action: Can be a list of desired motor angles for all motors when the
			robot is in position control mode; A list of desired motor torques. Or a
			list of tuples (q, qdot, kp, kd, tau) for hybrid control mode. The
			action must be compatible with the robot's motor control mode. Also, we
			are not going to use the leg space (swing/extension) definition at the
			gym level, since they are specific to Minitaur.

		Returns:
			observations: The observation dictionary. The keys are the sensor names
			and the values are the sensor readings.
			reward: The reward for the current state-action pair.
			done: Whether the episode has ended.
			info: A dictionary that stores diagnostic information.

		Raises:
			ValueError: The action dimension is not the same as the number of motors.
			ValueError: The magnitude of actions is out of bounds.
		"""
		self._last_base_position = self._robot.GetBasePosition()
		self._last_action = action

		if self._is_render:
			# Sleep, otherwise the computation takes less time than real time,
			# which will make the visualization like a fast-forward video.
			time_spent = time.time() - self._last_frame_time
			self._last_frame_time = time.time()
			time_to_sleep = self._env_time_step - time_spent
			if time_to_sleep > 0:
				time.sleep(time_to_sleep)
			base_pos = self._robot.GetBasePosition()

			# Also keep the previous orientation of the camera set by the user.
			[yaw, pitch, dist] = self._pybullet_client.getDebugVisualizerCamera()[8:11]
			self._pybullet_client.resetDebugVisualizerCamera(dist, yaw, pitch, base_pos)
			self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
			alpha = 1.
			if self._show_reference_id >= 0:
				alpha = self._pybullet_client.readUserDebugParameter(self._show_reference_id)

			ref_col = [1, 1, 1, alpha]
			if hasattr(self._task, '_ref_model'):
				self._pybullet_client.changeVisualShape(self._task._ref_model, -1, rgbaColor=ref_col)
				for l in range(self._pybullet_client.getNumJoints(self._task._ref_model)):
					self._pybullet_client.changeVisualShape(self._task._ref_model, l, rgbaColor=ref_col)

			# delay = self._pybullet_client.readUserDebugParameter(self._delay_id)
			# if (delay>0):
			# 	time.sleep(delay)

		# robot class and put the logics here.
		self._robot.Step(action)

		for s in self.all_sensors():
			s.on_step(self)

		if self._task and hasattr(self._task, 'update'):
			self._task.update(self)

		reward = self._reward()

		done = self._termination()
		self._env_step_counter += 1
		if done:
			self._robot.Terminate()
		return self._get_observation(), reward, done, {}

	def render(self, mode='rgb_array'):
		"""Renders the rgb view from the robots perspective.
			Currently tuned to get the view from the head of the robot"""

		# Base information
		proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(fov=66, aspect=1, nearVal=0.01, farVal=100)
		base_pos = list(self._robot.GetBasePosition())
		base_ori = list(self._robot.GetBaseOrientation())
		base_pos[2] = base_pos[2]+0.1

		# Rotate camera direction
		rot_mat = np.array(self._pybullet_client.getMatrixFromQuaternion(base_ori)).reshape(3, 3)
		camera_vec = np.matmul(rot_mat, [1, 0, 0]) # Target position is always the x-axis. This decides the diirection of the camera
		up_vec = np.matmul(rot_mat, np.array([0, 0, 1])) # Which direction is considered "up"
		view_matrix = self._pybullet_client.computeViewMatrix(base_pos, base_pos + camera_vec, up_vec)

		# if mode != 'rgb_array':
		# 	raise ValueError('Unsupported render mode:{}'.format(mode))
		# base_pos = self._robot.GetBasePosition()
		# # base_ori = self._robot.GetTrueBaseRollPitchYaw()
		# base_pos = [base_pos[0]+0.325, base_pos[1], base_pos[2]-0.03] # The true camera postion of the robot
		# view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
		# 	cameraTargetPosition=base_pos,
		# 	distance=1,
		# 	yaw=-90,
		# 	pitch=-10,
		# 	roll=0,
		# 	upAxisIndex=2)
		# proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(
		# 	fov=60,
		# 	aspect=float(self._render_width) / self._render_height,
		# 	nearVal=0.1,
		# 	farVal=100.0)
		(_, _, px, _, _) = self._pybullet_client.getCameraImage(
			width=self._render_width,
			height=self._render_height,
			renderer=self._pybullet_client.ER_BULLET_HARDWARE_OPENGL,
			viewMatrix=view_matrix,
			projectionMatrix=proj_matrix)
		rgb_array = np.array(px)
		rgb_array = rgb_array[:, :, :3]
		return rgb_array

	def close(self):
		"""Terminates the simulation"""
		if hasattr(self, '_robot') and self._robot:
			self._robot.Terminate()

	def seed(self, seed=None):
		self.np_random, self.np_random_seed = seeding.np_random(seed)
		return [self.np_random_seed]

	def _configureSimulator(self, gym_config):
		"""Configures pybullet with the provided settings"""
		self._on_rack = gym_config.robot_on_rack # Whether the robot is on rack or not
		self._is_render = gym_config.enable_rendering # Whether to render the GUI or not
		self._num_action_repeat = gym_config.num_action_repeat # The number of simulation steps that the same action is repeated.
		self._sim_time_step = gym_config.sim_time_step_s # The simulation time step 
		self._env_time_step = self._sim_time_step * self._num_action_repeat
		self._last_frame_time = 0.0 # The wall-clock time at which the last frame is rendered.
		self._show_reference_id = -1

		# Set the default render options.
		self._camera_dist = gym_config.camera_distance
		self._camera_yaw = gym_config.camera_yaw
		self._camera_pitch = gym_config.camera_pitch
		self._render_width = gym_config.render_width
		self._render_height = gym_config.render_height

		if self._num_action_repeat < 1:
			raise ValueError('number of action repeats should be at least 1.')

		# Render in GUI mode
		if self._is_render:
			self._pybullet_client = bullet_client.BulletClient(connection_mode=p.GUI)
			p.configureDebugVisualizer(p.COV_ENABLE_GUI, gym_config.enable_rendering_gui)
		# Render in DIRECT mode
		else:
			self._pybullet_client = bullet_client.BulletClient()

		self._pybullet_client.setAdditionalSearchPath(pd.getDataPath()) # Add the path to pybullet_data in the pybullet search path
		self._num_bullet_solver_iterations = int(_NUM_SIMULATION_ITERATION_STEPS / self._num_action_repeat)
		self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=int(np.round(self._num_bullet_solver_iterations)))
		self._pybullet_client.setTimeStep(self._sim_time_step)
		
		# Check if terrain need to be randomized
		self._randomise_terrain = gym_config.randomise_terrain
		if self._randomise_terrain:
			self._terrain_randomizer = TerrainRandomizer(self._pybullet_client)

		# using the eglRendererPlugin (hardware OpenGL acceleration)
		# using EGL on Linux and default OpenGL window on Win32.
		# if gym_config.egl_rendering:
		# 	egl = pkgutil.get_loader('eglRenderer')
		# 	if (egl):
		# 		self._pybullet_client.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
		# 	else:
		# 		self._pybullet_client.loadPlugin("eglRendererPlugin")

	def _build_action_space(self):
		"""Builds action space based on motor control mode."""
		motor_mode = self._gym_config.motor_control_mode
		action_upper_bound = []
		action_lower_bound = []
		action_config = a1.A1.ACTION_CONFIG
		if motor_mode == robot_config.MotorControlMode.HYBRID:
			for action in action_config:
				action_upper_bound.extend([6.28] * robot_config.HYBRID_ACTION_DIMENSION)
				action_lower_bound.extend([-6.28] * robot_config.HYBRID_ACTION_DIMENSION)
			self.action_space = spaces.Box(np.array(action_lower_bound),
											np.array(action_upper_bound),
											dtype=np.float32)
			import pdb; pdb.set_trace()
		elif motor_mode == robot_config.MotorControlMode.TORQUE:
			# TODO (yuxiangy): figure out the torque limits of robots.
			torque_limits = np.array([100] * len(action_config))
			self.action_space = spaces.Box(-torque_limits,
											torque_limits,
											dtype=np.float32)
		else:
			# Position mode
			for action in action_config:
				action_upper_bound.append(action.upper_bound)
				action_lower_bound.append(action.lower_bound)

			self.action_space = spaces.Box(np.array(action_lower_bound),
											np.array(action_upper_bound),
											dtype=np.float32)

	def _build_observation_space(self, robot_sensors, env_sensors=None):
		"""Builds the action space using the different sensors and ranges

		Args:
			robot_sensors: A list of sensors that are from the robot.
			env_sensors: A list of sensors that are from the environment
		"""
		self._robot_sensors = robot_sensors
		self._sensors = env_sensors if env_sensors is not None else list()

		# Construct the observation space from the list of sensors. Note that we
		# will reconstruct the observation_space after the robot is created.
		self.observation_space = space_utils.convert_sensors_to_gym_space_dictionary(self.all_sensors())
		self.observation_space = space_utils.flatten_observation_spaces(self.observation_space)
		if isinstance(self.observation_space, gym.spaces.Dict):
			self.observation_space = self.observation_space["others"]
		# self.observation_space = space_utils.convert_1d_box_sensors_to_gym_space(self.all_sensors())

	
	def all_sensors(self):
		"""Returns all robot and environmental sensors."""
		return self._robot_sensors + self._sensors

	def sensor_by_name(self, name):
		"""Returns the sensor with the given name, or None if not exist."""
		for sensor_ in self.all_sensors():
			if sensor_.get_name() == name:
				return sensor_
		return None

	def get_ground(self):
		"""Get simulation ground model."""
		return self._world_dict['ground']

	def set_ground(self, terrain_id):
		"""Set simulation ground model."""
		self._world_dict['ground'] = terrain_id

	def _termination(self):
		if not self._robot.is_safe:
			return True

		if self._task and hasattr(self._task, 'done'):
			return self._task.done(self)

		for s in self.all_sensors():
			s.on_terminate(self)

		return False

	def _reward(self):
		"""Calculates the reward for the task"""
		if self._task:
			return self._task(self)
		return 0

	def _get_observation(self):
		"""Get observation of this environment from a list of sensors.

		Returns:
			observations: sensory observation in the numpy array format
		"""
		sensors_dict = {}
		for s in self.all_sensors():
			sensors_dict[s.get_name()] = s.get_observation()

		observations = collections.OrderedDict(sorted(list(sensors_dict.items())))
		observations = space_utils.flatten_observation(observations)
		if isinstance(observations, dict):
			observations = observations["others"]
		return observations
		# observations = np.array([], dtype=np.float32)
		# for s in self.all_sensors():
		# 	observations = np.concatenate((observations, s.get_observation()))
		# return observations


	def get_time_since_reset(self):
		"""Get the time passed (in seconds) since the last reset.

		Returns:
			Time in seconds since the last reset.
		"""
		return self._robot.GetTimeSinceReset()

	@property
	def pybullet_client(self):
		return self._pybullet_client

	@property
	def robot(self):
		return self._robot

	@property
	def env_step_counter(self):
		return self._env_step_counter

	@property
	def hard_reset(self):
		return self._hard_reset

	@property
	def last_action(self):
		return self._last_action

	@property
	def env_time_step(self):
		return self._env_time_step
		
	@property
	def sim_time_step(self):
		return self._sim_time_step

	@property
	def task(self):
		return self._task

	@property
	def rendering_enabled(self):
		return self._is_render

	@property
	def last_base_position(self):
		return self._last_base_position

	@property
	def world_dict(self):
		return self._world_dict.copy()

	@world_dict.setter
	def world_dict(self, new_dict):
		self._world_dict = new_dict.copy()
