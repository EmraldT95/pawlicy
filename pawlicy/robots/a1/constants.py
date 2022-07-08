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

# Lint as: python3
"""Defines the laikago robot related constants and URDF specs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import pybullet as pyb # pytype: disable=import-error
import numpy as np

NUM_MOTORS = 12
NUM_LEGS = 4
MOTORS_PER_LEG = 3

INIT_RACK_POSITION = [0, 0, 1]
INIT_POSITION = [0, 0, 0.48]

# Will be default to (0, 0, 0, 1) once the new laikago_toes_zup.urdf checked in.
INIT_ORIENTATION = pyb.getQuaternionFromEuler([math.pi / 2.0, 0, math.pi / 2.0])

# Can be different from the motors, although for laikago they are the same list.
JOINT_NAMES = [
    "FR_hip_joint",
    "FR_upper_joint",
    "FR_lower_joint",
    "FL_hip_joint",
    "FL_upper_joint",
    "FL_lower_joint",
    "RR_hip_joint",
    "RR_upper_joint",
    "RR_lower_joint",
    "RL_hip_joint",
    "RL_upper_joint",
    "RL_lower_joint",
]

# INIT_ABDUCTION_ANGLE = 0
# INIT_HIP_ANGLE = 0.67
# INIT_KNEE_ANGLE = -1.25
INIT_ABDUCTION_ANGLE = 0
INIT_HIP_ANGLE = 0.9
INIT_KNEE_ANGLE = -1.8

# Note this matches the Laikago SDK/control convention, but is different from
# URDF's internal joint angles which needs to be computed using the joint
# offsets and directions. The conversion formula is (sdk_joint_angle + offset) *
# joint direction.
# INIT_JOINT_ANGLES = collections.OrderedDict(
#     zip(JOINT_NAMES,
#         (INIT_ABDUCTION_ANGLE, INIT_HIP_ANGLE, INIT_KNEE_ANGLE) * NUM_LEGS))

# Bases on the readings from Laikago's default pose.
INIT_MOTOR_ANGLES = np.array(
    [INIT_ABDUCTION_ANGLE, INIT_HIP_ANGLE, INIT_KNEE_ANGLE] * NUM_LEGS)

# Used to convert the robot SDK joint angles to URDF joint angles.
JOINT_DIRECTIONS = collections.OrderedDict(
    zip(JOINT_NAMES, (-1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1)))

# HIP_JOINT_OFFSET = 0.0
# UPPER_LEG_JOINT_OFFSET = -0.6
# KNEE_JOINT_OFFSET = 0.66
HIP_JOINT_OFFSET = 0.0
UPPER_LEG_JOINT_OFFSET = 0.0
KNEE_JOINT_OFFSET = 0.0

# Used to convert the robot SDK joint angles to URDF joint angles.
# JOINT_OFFSETS = collections.OrderedDict(
#     zip(JOINT_NAMES,
#         [HIP_JOINT_OFFSET, UPPER_LEG_JOINT_OFFSET, KNEE_JOINT_OFFSET] *
#         NUM_LEGS))

JOINT_OFFSETS = np.array(
    [HIP_JOINT_OFFSET, UPPER_LEG_JOINT_OFFSET, KNEE_JOINT_OFFSET] * NUM_LEGS)

LEG_NAMES = (
    "front_right",
    "front_left",
    "rear_right",
    "rear_left",
)

LEG_ORDER = (
    "front_right",
    "front_left",
    "back_right",
    "back_left",
)

END_EFFECTOR_NAMES = (
    "jtoeFR",
    "jtoeFL",
    "jtoeRR",
    "jtoeRL",
)

MOTOR_NAMES = JOINT_NAMES
MOTOR_GROUP = collections.OrderedDict((
    (LEG_NAMES[0], JOINT_NAMES[0:3]),
    (LEG_NAMES[1], JOINT_NAMES[3:6]),
    (LEG_NAMES[2], JOINT_NAMES[6:9]),
    (LEG_NAMES[3], JOINT_NAMES[9:12]),
))

# Regulates the joint angle change when in position control mode.
# MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 0.12
MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 0.2

# The hip joint location in the CoM frame.
# HIP_POSITIONS = collections.OrderedDict((
#     (LEG_NAMES[0], (0.21, -0.1157, 0)),
#     (LEG_NAMES[1], (0.21, 0.1157, 0)),
#     (LEG_NAMES[2], (-0.21, -0.1157, 0)),
#     (LEG_NAMES[3], (-0.21, 0.1157, 0)),
# ))
_DEFAULT_HIP_POSITIONS = (
    (0.17, -0.135, 0),
    (0.17, 0.13, 0),
    (-0.195, -0.135, 0),
    (-0.195, 0.13, 0),
)

COM_OFFSET = -np.array([0.012731, 0.002186, 0.000515])
HIP_OFFSETS = np.array([[0.183, -0.047, 0.], [0.183, 0.047, 0.],
                        [-0.183, -0.047, 0.], [-0.183, 0.047, 0.]
                        ]) + COM_OFFSET

SENSOR_NOISE_STDDEV = (0.0, 0.0, 0.0, 0.0, 0.0)

ABDUCTION_P_GAIN = 100.0
ABDUCTION_D_GAIN = 1.
HIP_P_GAIN = 100.0
HIP_D_GAIN = 2.0
KNEE_P_GAIN = 100.0
KNEE_D_GAIN = 2.0

MOTOR_KP = [
    ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
    ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
    ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
    ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN
]
MOTOR_KD = [
    ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
    ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
    ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
    ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN
]

URDF_FILEPATH = "a1/a1.urdf"

_BODY_B_FIELD_NUMBER = 2
_LINK_A_FIELD_NUMBER = 3