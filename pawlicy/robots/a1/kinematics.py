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

"""The inverse kinematic utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import typing
from . import constants

_IDENTITY_ORIENTATION = (0, 0, 0, 1)


def joint_angles_from_link_position(
    robot: typing.Any,
    link_position: typing.Sequence[float],
    link_id: int,
    joint_ids: typing.Sequence[int],
    base_translation: typing.Sequence[float] = (0, 0, 0),
    base_rotation: typing.Sequence[float] = (0, 0, 0, 1)):
  """Uses Inverse Kinematics to calculate joint angles.

  Args:
    robot: A robot instance.
    link_position: The (x, y, z) of the link in the body frame. This local frame
      is transformed relative to the COM frame using a given translation and
      rotation.
    link_id: The link id as returned from loadURDF.
    joint_ids: The positional index of the joints. This can be different from
      the joint unique ids.
    base_translation: Additional base translation.
    base_rotation: Additional base rotation.

  Returns:
    A list of joint angles.
  """
  # Projects to local frame.
  base_position, base_orientation = robot.GetBasePosition(
  ), robot.GetBaseOrientation()
  base_position, base_orientation = robot.pybullet_client.multiplyTransforms(
      base_position, base_orientation, base_translation, base_rotation)

  # Projects to world space.
  world_link_pos, _ = robot.pybullet_client.multiplyTransforms(
      base_position, base_orientation, link_position, _IDENTITY_ORIENTATION)
  ik_solver = 0
  all_joint_angles = robot.pybullet_client.calculateInverseKinematics(
      robot.quadruped, link_id, world_link_pos, solver=ik_solver)

  # Extract the relevant joint angles.
  joint_angles = [all_joint_angles[i] for i in joint_ids]
  return joint_angles


def link_position_in_base_frame(
    robot: typing.Any,
    link_id: int,
):
  """Computes the link's local position in the robot frame.

  Args:
    robot: A robot instance.
    link_id: The link to calculate its relative position.

  Returns:
    The relative position of the link.
  """
  base_position, base_orientation = robot.GetBasePosition(
  ), robot.GetBaseOrientation()
  inverse_translation, inverse_rotation = robot.pybullet_client.invertTransform(
      base_position, base_orientation)

  link_state = robot.pybullet_client.getLinkState(robot.quadruped, link_id)
  link_position = link_state[0]
  link_local_position, _ = robot.pybullet_client.multiplyTransforms(
      inverse_translation, inverse_rotation, link_position, (0, 0, 0, 1))

  return np.array(link_local_position)


def compute_jacobian(
    robot: typing.Any,
    link_id: int,
):
  """Computes the Jacobian matrix for the given link.

  Args:
    robot: A robot instance.
    link_id: The link id as returned from loadURDF.

  Returns:
    The 3 x N transposed Jacobian matrix. where N is the total DoFs of the
    robot. For a quadruped, the first 6 columns of the matrix corresponds to
    the CoM translation and rotation. The columns corresponds to a leg can be
    extracted with indices [6 + leg_id * 3: 6 + leg_id * 3 + 3].
  """

  all_joint_angles = [state[0] for state in robot.joint_states]
  zero_vec = [0] * len(all_joint_angles)
  jv, _ = robot.pybullet_client.calculateJacobian(robot.quadruped, link_id,
                                                  (0, 0, 0), all_joint_angles,
                                                  zero_vec, zero_vec)
  jacobian = np.array(jv)
  return jacobian


def foot_position_in_hip_frame_to_joint_angle(foot_position, l_hip_sign=1):
    l_up = 0.2
    l_low = 0.2
    l_hip = 0.08505 * l_hip_sign
    x, y, z = foot_position[0], foot_position[1], foot_position[2]
    theta_knee = -np.arccos(
        (x**2 + y**2 + z**2 - l_hip**2 - l_low**2 - l_up**2) /
        (2 * l_low * l_up))
    l = np.sqrt(l_up**2 + l_low**2 + 2 * l_up * l_low * np.cos(theta_knee))
    theta_hip = np.arcsin(-x / l) - theta_knee / 2
    c1 = l_hip * y - l * np.cos(theta_hip + theta_knee / 2) * z
    s1 = l * np.cos(theta_hip + theta_knee / 2) * y + l_hip * z
    theta_ab = np.arctan2(s1, c1)
    return np.array([theta_ab, theta_hip, theta_knee])


def foot_position_in_hip_frame(angles, l_hip_sign=1):
    theta_ab, theta_hip, theta_knee = angles[0], angles[1], angles[2]
    l_up = 0.2
    l_low = 0.2
    l_hip = 0.08505 * l_hip_sign
    leg_distance = np.sqrt(l_up**2 + l_low**2 +
                           2 * l_up * l_low * np.cos(theta_knee))
    eff_swing = theta_hip + theta_knee / 2

    off_x_hip = -leg_distance * np.sin(eff_swing)
    off_z_hip = -leg_distance * np.cos(eff_swing)
    off_y_hip = l_hip

    off_x = off_x_hip
    off_y = np.cos(theta_ab) * off_y_hip - np.sin(theta_ab) * off_z_hip
    off_z = np.sin(theta_ab) * off_y_hip + np.cos(theta_ab) * off_z_hip
    return np.array([off_x, off_y, off_z])


def analytical_leg_jacobian(leg_angles, leg_id):
    """
    Computes the analytical Jacobian.
    Args:
    ` leg_angles: a list of 3 numbers for current abduction, hip and knee angle.
      l_hip_sign: whether it's a left (1) or right(-1) leg.
    """
    l_up = 0.2
    l_low = 0.2
    l_hip = 0.08505 * (-1)**(leg_id + 1)

    t1, t2, t3 = leg_angles[0], leg_angles[1], leg_angles[2]
    l_eff = np.sqrt(l_up**2 + l_low**2 + 2 * l_up * l_low * np.cos(t3))
    t_eff = t2 + t3 / 2
    J = np.zeros((3, 3))
    J[0, 0] = 0
    J[0, 1] = -l_eff * np.cos(t_eff)
    J[0, 2] = l_low * l_up * np.sin(t3) * np.sin(t_eff) / l_eff - l_eff * np.cos(
        t_eff) / 2
    J[1, 0] = -l_hip * np.sin(t1) + l_eff * np.cos(t1) * np.cos(t_eff)
    J[1, 1] = -l_eff * np.sin(t1) * np.sin(t_eff)
    J[1, 2] = -l_low * l_up * np.sin(t1) * np.sin(t3) * np.cos(
        t_eff) / l_eff - l_eff * np.sin(t1) * np.sin(t_eff) / 2
    J[2, 0] = l_hip * np.cos(t1) + l_eff * np.sin(t1) * np.cos(t_eff)
    J[2, 1] = l_eff * np.sin(t_eff) * np.cos(t1)
    J[2, 2] = l_low * l_up * np.sin(t3) * np.cos(t1) * np.cos(
        t_eff) / l_eff + l_eff * np.sin(t_eff) * np.cos(t1) / 2
    return J


# For JIT compilation
foot_position_in_hip_frame_to_joint_angle(np.random.uniform(size=3), 1)
foot_position_in_hip_frame_to_joint_angle(np.random.uniform(size=3), -1)


def foot_positions_in_base_frame(foot_angles):
    foot_angles = foot_angles.reshape((4, 3))
    foot_positions = np.zeros((4, 3))
    for i in range(4):
        foot_positions[i] = foot_position_in_hip_frame(foot_angles[i],
                                                       l_hip_sign=(-1)**(i + 1))
    return foot_positions + constants.HIP_OFFSETS