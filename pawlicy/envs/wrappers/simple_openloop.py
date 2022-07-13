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

"""Simple openloop trajectory generators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import attr
from gym import spaces
import numpy as np

from pawlicy.robots import laikago_pose_utils

class LaikagoPoseOffsetGenerator(object):
  """A trajectory generator that return constant motor angles."""
  def __init__(
      self,
      init_abduction=laikago_pose_utils.LAIKAGO_DEFAULT_ABDUCTION_ANGLE,
      init_hip=laikago_pose_utils.LAIKAGO_DEFAULT_HIP_ANGLE,
      init_knee=laikago_pose_utils.LAIKAGO_DEFAULT_KNEE_ANGLE,
      action_limit=0.5,
  ):
    """Initializes the controller.
    Args:
      action_limit: a tuple of [limit_abduction, limit_hip, limit_knee]
    """
    self._pose = np.array(
        attr.astuple(
            laikago_pose_utils.LaikagoPose(abduction_angle_0=init_abduction,
                                           hip_angle_0=init_hip,
                                           knee_angle_0=init_knee,
                                           abduction_angle_1=init_abduction,
                                           hip_angle_1=init_hip,
                                           knee_angle_1=init_knee,
                                           abduction_angle_2=init_abduction,
                                           hip_angle_2=init_hip,
                                           knee_angle_2=init_knee,
                                           abduction_angle_3=init_abduction,
                                           hip_angle_3=init_hip,
                                           knee_angle_3=init_knee)))
    action_high = np.array([action_limit] * 12)
    self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)

  def reset(self):
    pass

  def get_action(self, current_time=None, input_action=None):
    """Computes the trajectory according to input time and action.
    Args:
      current_time: The time in gym env since reset.
      input_action: A numpy array. The input leg pose from a NN controller.
    Returns:
      A numpy array. The desired motor angles.
    """
    del current_time
    return self._pose + input_action

  def get_observation(self, input_observation):
    """Get the trajectory generator's observation."""

    return input_observation


class SimpleRobotOffsetGenerator(object):
    """A trajectory generator that return constant motor angles."""

    def __init__(
            self,
            pose,
            action_limit=0.5,
    ):
        """Initializes the controller.
      Args:
        action_limit: a tuple of [limit_abduction, limit_hip, limit_knee]
      """
        self._pose = np.array(
            attr.astuple(
                laikago_pose_utils.LaikagoPose(abduction_angle_0=pose[0],
                                               hip_angle_0=pose[1],
                                               knee_angle_0=pose[2],
                                               abduction_angle_1=pose[3],
                                               hip_angle_1=pose[4],
                                               knee_angle_1=pose[5],
                                               abduction_angle_2=pose[6],
                                               hip_angle_2=pose[7],
                                               knee_angle_2=pose[8],
                                               abduction_angle_3=pose[9],
                                               hip_angle_3=pose[10],
                                               knee_angle_3=pose[11])))
        action_high = np.array([action_limit] * 12)
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)

    def reset(self):
        pass

    def get_action(self, current_time=None, input_action=None):
        """Computes the trajectory according to input time and action.
        Args:
          current_time: The time in gym env since reset.
          input_action: A numpy array. The input leg pose from a NN controller.
        Returns:
          A numpy array. The desired motor angles.
        """
        del current_time
        return self._pose + input_action

    def get_observation(self, input_observation):
        """Get the trajectory generator's observation."""

        return input_observation
