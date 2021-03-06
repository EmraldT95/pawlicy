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
"""An env wrapper that flattens the observation dictionary to an array."""
import collections
import gym
import numpy as np


class ObservationDictionaryToArrayWrapper(gym.Env):
	"""An env wrapper that flattens the observation dictionary to an array."""

	def __init__(self, gym_env, observation_excluded=()):
		"""Initializes the wrapper."""
		self.observation_excluded = observation_excluded
		self._gym_env = gym_env
		self.observation_space = self._flatten_observation_spaces(
			self._gym_env.observation_space, self.observation_excluded)
		self.action_space = self._gym_env.action_space

	def __getattr__(self, attr):
		return getattr(self._gym_env, attr)

	def _flatten_observation_spaces(self, observation_spaces, observation_excluded):
		"""Flattens the dictionary observation spaces to gym.spaces.Box.

		If observation_excluded is passed in, it will still return a dictionary,
		which includes all the (key, observation_spaces[key]) in observation_excluded,
		and ('other': the flattened Box space).

		Args:
			observation_spaces: A dictionary of all the observation spaces.
			observation_excluded: A list/tuple of all the keys of the observations to be
				ignored during flattening.

		Returns:
			A box space or a dictionary of observation spaces based on whether
				observation_excluded is empty.
		"""
		if not isinstance(observation_excluded, (list, tuple)):
			observation_excluded = [observation_excluded]
		lower_bound = []
		upper_bound = []
		for key, value in observation_spaces.spaces.items():
			if key not in observation_excluded:
				lower_bound.append(np.asarray(value.low).flatten())
				upper_bound.append(np.asarray(value.high).flatten())
		lower_bound = np.concatenate(lower_bound)
		upper_bound = np.concatenate(upper_bound)
		observation_space = gym.spaces.Box(
				np.array(lower_bound), np.array(upper_bound), dtype=np.float32)
		if not observation_excluded:
			return observation_space
		else:
			observation_spaces_after_flatten = {"other": observation_space}
			for key in observation_excluded:
				observation_spaces_after_flatten[key] = observation_spaces[key]
			return gym.spaces.Dict(observation_spaces_after_flatten)

	def _flatten_observation(self, observation_dict, observation_excluded):
		"""Flattens the observation dictionary to an array.

		If observation_excluded is passed in, it will still return a dictionary,
		which includes all the (key, observation_dict[key]) in observation_excluded,
		and ('other': the flattened array).

		Args:
			observation_dict: A dictionary of all the observations.
			observation_excluded: A list/tuple of all the keys of the observations to be
				ignored during flattening.

		Returns:
			An array or a dictionary of observations based on whether
				observation_excluded is empty.
		"""
		if not isinstance(observation_excluded, (list, tuple)):
			observation_excluded = [observation_excluded]
		observations = []
		for key, value in observation_dict.items():
			if key not in observation_excluded:
				observations.append(np.asarray(value).flatten())
		flat_observations = np.concatenate(observations)
		if not observation_excluded:
			return flat_observations
		else:
			observation_dict_after_flatten = {"other": flat_observations}
			for key in observation_excluded:
				observation_dict_after_flatten[key] = observation_dict[key]
			return collections.OrderedDict(sorted(list(observation_dict_after_flatten.items())))

	def reset(self, initial_motor_angles=None, reset_duration=0.0):
		observation = self._gym_env.reset(
			initial_motor_angles=initial_motor_angles,
			reset_duration=reset_duration)
		return self._flatten_observation(observation)

	def step(self, action):
		"""Steps the wrapped environment.

		Args:
			action: Numpy array. The input action from an NN agent.

		Returns:
			The tuple containing the flattened observation, the reward, the epsiode
				end indicator.
		"""
		observation_dict, reward, done, _ = self._gym_env.step(action)
		return self._flatten_observation(observation_dict, self.observation_excluded), reward, done, _

	def render(self, mode='human'):
		return self._gym_env.render(mode)
