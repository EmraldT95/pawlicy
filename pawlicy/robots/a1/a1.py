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

"""Pybullet simulation of a Laikago robot."""


from pawlicy.envs import gym_config
from pawlicy.robots import robot_config
from pawlicy.robots.a1 import kinematics, motor
from pawlicy.robots.a1 import constants
from pawlicy.robots.a1 import action_filter

import pybullet as pyb  # pytype: disable=import-error
import numpy as np

import copy
import collections
import re
import math

INIT_RACK_POSITION = [0, 0, 1]
INIT_ORIENTATION = [0, 0, 0, 1]
JOINT_DIRECTIONS = np.ones(12)

HIP_NAME_PATTERN = re.compile(r"\w+_hip_\w+")
UPPER_NAME_PATTERN = re.compile(r"\w+_upper_\w+")
LOWER_NAME_PATTERN = re.compile(r"\w+_lower_\w+")
TOE_NAME_PATTERN = re.compile(r"\w+_toe\d*")
IMU_NAME_PATTERN = re.compile(r"imu\d*")
OVERHEAT_SHUTDOWN_TORQUE = 2.45
OVERHEAT_SHUTDOWN_TIME = 1.0

def MapToMinusPiToPi(angles):
    """Maps a list of angles to [-pi, pi].

    Args:
      angles: A list of angles in rad.

    Returns:
      A list of angle mapped to [-pi, pi].
    """
    mapped_angles = copy.deepcopy(angles)
    for i in range(len(angles)):
        mapped_angles[i] = math.fmod(angles[i], 2 * math.pi)
        if mapped_angles[i] >= math.pi:
            mapped_angles[i] -= 2 * math.pi
        elif mapped_angles[i] < -math.pi:
            mapped_angles[i] += 2 * math.pi
    return mapped_angles


class A1(object):
    """A simulation for the Laikago robot."""

    # At high replanning frequency, inaccurate values of BODY_MASS/INERTIA
    # doesn't seem to matter much. However, these values should be better tuned
    # when the replan frequency is low (e.g. using a less beefy CPU).
    MPC_BODY_MASS = 108 / 9.8
    MPC_BODY_INERTIA = np.array((0.017, 0, 0, 0, 0.057, 0, 0, 0, 0.064)) * 4.
    MPC_BODY_HEIGHT = 0.24
    MPC_VELOCITY_MULTIPLIER = 0.5
    ACTION_CONFIG = [
        gym_config.ScalarField(name="FR_hip_motor", upper_bound=0.802851455917, lower_bound=-0.802851455917),
        gym_config.ScalarField(name="FR_upper_joint", upper_bound=4.18879020479, lower_bound=-1.0471975512),
        gym_config.ScalarField(name="FR_lower_joint", upper_bound=-0.916297857297, lower_bound=-2.69653369433),
        gym_config.ScalarField(name="FL_hip_motor", upper_bound=0.802851455917, lower_bound=-0.802851455917),
        gym_config.ScalarField(name="FL_upper_joint", upper_bound=4.18879020479, lower_bound=-1.0471975512),
        gym_config.ScalarField(name="FL_lower_joint", upper_bound=-0.916297857297, lower_bound=-2.69653369433),
        gym_config.ScalarField(name="RR_hip_motor", upper_bound=0.802851455917, lower_bound=-0.802851455917),
        gym_config.ScalarField(name="RR_upper_joint", upper_bound=4.18879020479, lower_bound=-1.0471975512),
        gym_config.ScalarField(name="RR_lower_joint", upper_bound=-0.916297857297, lower_bound=-2.69653369433),
        gym_config.ScalarField(name="RL_hip_motor", upper_bound=0.802851455917, lower_bound=-0.802851455917),
        gym_config.ScalarField(name="RL_upper_joint", upper_bound=4.18879020479, lower_bound=-1.0471975512),
        gym_config.ScalarField(name="RL_lower_joint", upper_bound=-0.916297857297, lower_bound=-2.69653369433),
    ]

    def __init__(
        self,
        pybullet_client,
        enable_clip_motor_commands=False,
        time_step=0.001,
        action_repeat=10,
        sensors=None,
        control_latency=0.002,
        on_rack=False,
        enable_action_interpolation=True,
        enable_action_filter=False,
        reset_time=1,
        allow_knee_contact=False,
        self_collision_enabled=False,
        motor_control_mode=robot_config.MotorControlMode.POSITION,
        motor_torque_limits=None,
        pd_latency=0.0,
        observation_noise_stdev=constants.SENSOR_NOISE_STDDEV,
        motor_overheat_protection=False,
        motor_direction=JOINT_DIRECTIONS,
        motor_offset=constants.JOINT_OFFSETS,
        reset_at_current_position=False,
        init_position=INIT_RACK_POSITION
    ):
        self._pybullet_client = pybullet_client
        self._enable_clip_motor_commands = enable_clip_motor_commands
        self._action_repeat = action_repeat
        self.SetAllSensors(sensors if sensors is not None else list())
        self._control_latency = control_latency
        self._on_rack = on_rack
        self._enable_action_interpolation = enable_action_interpolation
        self._enable_action_filter = enable_action_filter
        self._allow_knee_contact = allow_knee_contact
        self._self_collision_enabled = self_collision_enabled
        self._motor_direction = motor_direction
        self._motor_offset = motor_offset
        self._pd_latency = pd_latency
        self._observation_noise_stdev = observation_noise_stdev
        self._motor_overheat_protection = motor_overheat_protection
        self._reset_at_current_position = reset_at_current_position
        self.time_step = time_step

        
        # This will be overwritten based on the terrain loaded
        self.init_position = init_position
        self._observed_motor_torques = np.zeros(constants.NUM_MOTORS)
        self._applied_motor_torques = np.zeros(constants.NUM_MOTORS)
        self._max_force = 3.5
        self._observation_history = collections.deque(maxlen=100)
        self._control_observation = []
        self._chassis_link_ids = [-1]
        self._leg_link_ids = []
        self._motor_link_ids = []
        self._foot_link_ids = []
        self._is_safe = True
        self._last_action = None
        self._motor_kps = np.asarray(constants.MOTOR_KP)
        self._motor_kds = np.asarray(constants.MOTOR_KD)
        self._step_counter = 0
        self._state_action_counter = 0 # This also includes the time spent during the Reset motion.

        if isinstance(motor_torque_limits, (collections.Sequence, np.ndarray)):
            self._motor_torque_limits = np.asarray(motor_torque_limits)
        elif motor_torque_limits is None:
            self._motor_torque_limits = None
        else:
            self._motor_torque_limits = motor_torque_limits

        self._motor_control_mode = motor_control_mode
        self._motor_model = motor.LaikagoMotorModel(
            kp=constants.MOTOR_KP,
            kd=constants.MOTOR_KD,
            torque_limits=self._motor_torque_limits,
            motor_control_mode=motor_control_mode)

        _, self._init_orientation_inv = self._pybullet_client.invertTransform(position=[0, 0, 0], orientation=INIT_ORIENTATION)

        if self._enable_action_filter:
            self._action_filter = self._BuildActionFilter()

        if self._on_rack and self._reset_at_current_position:
            raise ValueError("on_rack and reset_at_current_position cannot be enabled together")

        # reset_time=-1.0 means skipping the reset motion.
        # See Reset for more details.
        self.Reset(reset_time=reset_time)
        self.ReceiveObservation()

    def Reset(self, reload_urdf=True, default_motor_angles=None, reset_time=3.0):
        """Reset the minitaur to its initial states.

        Args:
        reload_urdf: Whether to reload the urdf file. If not, Reset() just place
            the minitaur back to its starting position.
        default_motor_angles: The default motor angles. If it is None, minitaur
            will hold a default pose (motor angle math.pi / 2) for 100 steps. In
            torque control mode, the phase of holding the default pose is skipped.
        reset_time: The duration (in seconds) to hold the default motor angles. If
            reset_time <= 0 or in torque control mode, the phase of holding the
            default pose is skipped.
        """
        if reload_urdf:
            self._LoadRobotURDF()
            self._BuildJointNameToIdDict()
            self._BuildUrdfIds()
            self._RemoveDefaultJointDamping()
            self._BuildMotorIdList()
            self._RecordMassInfoFromURDF()
            self._RecordInertiaInfoFromURDF()
            self.ResetPose(add_constraint=True)
        else:
            self._pybullet_client.resetBasePositionAndOrientation(self.quadruped, self.init_position, INIT_ORIENTATION)
            self._pybullet_client.resetBaseVelocity(self.quadruped, [0, 0, 0], [0, 0, 0])
            self.ResetPose(add_constraint=False)

        self._overheat_counter = np.zeros(constants.NUM_MOTORS)
        self._motor_enabled_list = [True] * constants.NUM_MOTORS
        self._observation_history.clear()
        self._step_counter = 0
        self._state_action_counter = 0
        self._is_safe = True
        self._last_action = None

        self._SettleDownForReset(default_motor_angles, reset_time)
        if self._enable_action_filter:
            self._action_filter.reset()

    def Terminate(self):
        pass

    def _LoadRobotURDF(self):
        if self._self_collision_enabled:
            self.quadruped = self._pybullet_client.loadURDF(
                constants.URDF_FILEPATH,
                self.init_position,
                INIT_ORIENTATION,
                useFixedBase=self._on_rack,
                flags=self._pybullet_client.URDF_USE_SELF_COLLISION)
        else:
            self.quadruped = self._pybullet_client.loadURDF(
                constants.URDF_FILEPATH,
                self.init_position,
                INIT_ORIENTATION,
                useFixedBase=self._on_rack)

    def _SettleDownForReset(self, default_motor_angles, reset_time):
        self.ReceiveObservation()
        if reset_time <= 0:
            return

        for _ in range(500):
            self._StepInternal(
                constants.INIT_MOTOR_ANGLES,
                motor_control_mode=robot_config.MotorControlMode.POSITION)

        if default_motor_angles is not None:
            num_steps_to_reset = int(reset_time / self.time_step)
            for _ in range(num_steps_to_reset):
                self._StepInternal(
                    default_motor_angles,
                    motor_control_mode=robot_config.MotorControlMode.POSITION)

    def _StepInternal(self, action, motor_control_mode):
        self.ApplyAction(action, motor_control_mode)
        # self._pybullet_client.setRealTimeSimulation(1)
        self._pybullet_client.stepSimulation()
        self.ReceiveObservation()
        self._state_action_counter += 1

    def Step(self, action, control_mode=None):
        """Steps simulation."""
        if self._enable_action_filter:
            action = self._FilterAction(action)
        if control_mode==None:
            control_mode = self._motor_control_mode
        for i in range(self._action_repeat):
            proc_action = self.ProcessAction(action, i)
            self._StepInternal(proc_action, control_mode)
            self._step_counter += 1
        self._last_action = action

    def GetHipPositionsInBaseFrame(self):
        return constants._DEFAULT_HIP_POSITIONS

    def GetFootContacts(self):
        all_contacts = self._pybullet_client.getContactPoints(bodyA=self.quadruped)

        contacts = [False, False, False, False]
        for contact in all_contacts:
            # Ignore self contacts
            if contact[constants._BODY_B_FIELD_NUMBER] == self.quadruped:
                continue
            try:
                toe_link_index = self._foot_link_ids.index(contact[constants._LINK_A_FIELD_NUMBER])
                contacts[toe_link_index] = True
            except ValueError:
                continue

        return contacts

    def ResetPose(self, add_constraint):
        del add_constraint
        for name in self._joint_name_to_id:
            joint_id = self._joint_name_to_id[name]
            self._pybullet_client.setJointMotorControl2(
                bodyIndex=self.quadruped,
                jointIndex=(joint_id),
                controlMode=self._pybullet_client.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0)
        for name, i in zip(constants.JOINT_NAMES, range(len(constants.JOINT_NAMES))):
            if "hip_joint" in name:
                angle = constants.INIT_MOTOR_ANGLES[i] + constants.HIP_JOINT_OFFSET
            elif "upper_joint" in name:
                angle = constants.INIT_MOTOR_ANGLES[i] + constants.UPPER_LEG_JOINT_OFFSET
            elif "lower_joint" in name:
                angle = constants.INIT_MOTOR_ANGLES[i] + constants.KNEE_JOINT_OFFSET
            else:
                raise ValueError("The name %s is not recognized as a motor joint." % name)
            self._pybullet_client.resetJointState(self.quadruped,
                                                  self._joint_name_to_id[name],
                                                  angle,
                                                  targetVelocity=0)

    def GetTimeSinceReset(self):
        return self._step_counter * self.time_step

    def _BuildActionFilter(self):
        sampling_rate = 1 / (self.time_step * self._action_repeat)
        a_filter = action_filter.ActionFilterButter(sampling_rate=sampling_rate, num_joints=constants.NUM_MOTORS)
        return a_filter

    def _ResetActionFilter(self):
        self._action_filter.reset()

    def _FilterAction(self, action):
        # initialize the filter history, since resetting the filter will fill
        # the history with zeros and this can cause sudden movements at the start
        # of each episode
        if self._step_counter == 0:
            default_action = self.GetMotorAngles()
            self._action_filter.init_history(default_action)

        filtered_action = self._action_filter.filter(action)
        return filtered_action

    def ProcessAction(self, action, substep_count):
        """If enabled, interpolates between the current and previous actions.

        Args:
        action: current action.
        substep_count: the step count should be between [0, self.__action_repeat).

        Returns:
        If interpolation is enabled, returns interpolated action depending on
        the current action repeat substep.
        """
        if self._enable_action_interpolation and self._last_action is not None:
            lerp = float(substep_count + 1) / self._action_repeat
            proc_action = self._last_action + lerp * (action - self._last_action)
        else:
            proc_action = action

        return proc_action

    def _BuildJointNameToIdDict(self):
        num_joints = self._pybullet_client.getNumJoints(self.quadruped)
        self._joint_name_to_id = {}
        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
            self._joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]

    def _BuildUrdfIds(self):
        """Build the link Ids from its name in the URDF file.

        Raises:
          ValueError: Unknown category of the joint name.
        """
        num_joints = self._pybullet_client.getNumJoints(self.quadruped)
        self._hip_link_ids = []
        self._leg_link_ids = []
        self._motor_link_ids = []
        self._lower_link_ids = []
        self._foot_link_ids = []
        self._imu_link_ids = []

        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
            joint_name = joint_info[1].decode("UTF-8")
            joint_id = self._joint_name_to_id[joint_name]
            if HIP_NAME_PATTERN.match(joint_name):
                self._hip_link_ids.append(joint_id)
            elif UPPER_NAME_PATTERN.match(joint_name):
                self._motor_link_ids.append(joint_id)
            # We either treat the lower leg or the toe as the foot link, depending on
            # the urdf version used.
            elif LOWER_NAME_PATTERN.match(joint_name):
                self._lower_link_ids.append(joint_id)
            elif TOE_NAME_PATTERN.match(joint_name):
                self._foot_link_ids.append(joint_id)
            elif IMU_NAME_PATTERN.match(joint_name):
                self._imu_link_ids.append(joint_id)
            else:
                raise ValueError("Unknown category of joint %s" % joint_name)

        self._leg_link_ids.extend(self._lower_link_ids)
        self._leg_link_ids.extend(self._foot_link_ids)

        #assert len(self._foot_link_ids) == NUM_LEGS
        self._hip_link_ids.sort()
        self._motor_link_ids.sort()
        self._lower_link_ids.sort()
        self._foot_link_ids.sort()
        self._leg_link_ids.sort()

    def _RemoveDefaultJointDamping(self):
        num_joints = self._pybullet_client.getNumJoints(self.quadruped)
        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
            self._pybullet_client.changeDynamics(joint_info[0], -1, linearDamping=0, angularDamping=0)

    def _BuildMotorIdList(self):
        self._motor_id_list = [self._joint_name_to_id[motor_name] for motor_name in self._GetMotorNames()]

    def _RecordMassInfoFromURDF(self):
        """Records the mass information from the URDF file."""
        self._base_mass_urdf = []
        for chassis_id in self._chassis_link_ids:
            self._base_mass_urdf.append(self._pybullet_client.getDynamicsInfo(self.quadruped, chassis_id)[0])
        self._leg_masses_urdf = []
        for leg_id in self._leg_link_ids:
            self._leg_masses_urdf.append(self._pybullet_client.getDynamicsInfo(self.quadruped, leg_id)[0])
        for motor_id in self._motor_link_ids:
            self._leg_masses_urdf.append(self._pybullet_client.getDynamicsInfo(self.quadruped, motor_id)[0])

    def _RecordInertiaInfoFromURDF(self):
        """Record the inertia of each body from URDF file."""
        self._link_urdf = []
        num_bodies = self._pybullet_client.getNumJoints(self.quadruped)
        for body_id in range(-1, num_bodies):  # -1 is for the base link.
            inertia = self._pybullet_client.getDynamicsInfo(self.quadruped, body_id)[2]
            self._link_urdf.append(inertia)
        # We need to use id+1 to index self._link_urdf because it has the base
        # (index = -1) at the first element.
        self._base_inertia_urdf = [self._link_urdf[chassis_id + 1] for chassis_id in self._chassis_link_ids]
        self._leg_inertia_urdf = [self._link_urdf[leg_id + 1] for leg_id in self._leg_link_ids]
        self._leg_inertia_urdf.extend([self._link_urdf[motor_id + 1] for motor_id in self._motor_link_ids])

    def _GetMotorNames(self):
        return constants.JOINT_NAMES

    def GetDefaultInitJointPose(self):
        """Get default initial joint pose."""
        joint_pose = (constants.INIT_MOTOR_ANGLES + constants.JOINT_OFFSETS) * JOINT_DIRECTIONS
        return joint_pose

    def ApplyAction(self, motor_commands, motor_control_mode=None):
        """Clips and then apply the motor commands using the motor model.

        Args:
          motor_commands: np.array. Can be motor angles, torques, hybrid commands,
            or motor pwms (for Minitaur only).N
          motor_control_mode: A MotorControlMode enum.
        """
        if self._enable_clip_motor_commands:
            motor_commands = self._ClipMotorCommands(motor_commands)
        self.last_action_time = self._state_action_counter * self.time_step
        control_mode = motor_control_mode

        if control_mode is None:
            control_mode = self._motor_control_mode

        motor_commands = np.asarray(motor_commands)

        q, qdot = self._GetPDObservation()
        qdot_true = self.GetTrueMotorVelocities()
        actual_torque, observed_torque = self._motor_model.convert_to_torque(motor_commands, q, qdot, qdot_true, control_mode)

        # May turn off the motor
        self._ApplyOverheatProtection(actual_torque)

        # The torque is already in the observation space because we use
        # GetMotorAngles and GetMotorVelocities.
        self._observed_motor_torques = observed_torque

        # Transform into the motor space when applying the torque.
        self._applied_motor_torque = np.multiply(actual_torque, self._motor_direction)
        motor_ids = []
        motor_torques = []

        for motor_id, motor_torque, motor_enabled in zip(self._motor_id_list, self._applied_motor_torque, self._motor_enabled_list):
            if motor_enabled:
                motor_ids.append(motor_id)
                motor_torques.append(motor_torque)
            else:
                motor_ids.append(motor_id)
                motor_torques.append(0)
        self._SetMotorTorqueByIds(motor_ids, motor_torques)

    def _GetPDObservation(self):
        pd_delayed_observation = self._GetDelayedObservation(self._pd_latency)
        q = pd_delayed_observation[0:constants.NUM_MOTORS]
        qdot = pd_delayed_observation[constants.NUM_MOTORS:2 * constants.NUM_MOTORS]
        return (np.array(q), np.array(qdot))

    def _ApplyOverheatProtection(self, actual_torque):
        if self._motor_overheat_protection:
            for i in range(constants.NUM_MOTORS):
                if abs(actual_torque[i]) > OVERHEAT_SHUTDOWN_TORQUE:
                    self._overheat_counter[i] += 1
                else:
                    self._overheat_counter[i] = 0
                if (self._overheat_counter[i] > OVERHEAT_SHUTDOWN_TIME / self.time_step):
                    self._motor_enabled_list[i] = False

    def _SetMotorTorqueByIds(self, motor_ids, torques):
        self._pybullet_client.setJointMotorControlArray(
            bodyIndex=self.quadruped,
            jointIndices=motor_ids,
            controlMode=self._pybullet_client.TORQUE_CONTROL,
            forces=torques)

    def _ClipMotorCommands(self, motor_commands):
        """Clips motor commands.

        Args:
          motor_commands: np.array. Can be motor angles, torques, hybrid commands,
            or motor pwms (for Minitaur only).

        Returns:
          Clipped motor commands.
        """

        # clamp the motor command by the joint limit, in case weired things happens
        max_angle_change = constants.MAX_MOTOR_ANGLE_CHANGE_PER_STEP
        current_motor_angles = self.GetMotorAngles()
        motor_commands = np.clip(motor_commands,
                                 current_motor_angles - max_angle_change,
                                 current_motor_angles + max_angle_change)
        return motor_commands

    @classmethod
    def GetConstants(cls):
        del cls
        return constants

    def ComputeMotorAnglesFromFootLocalPosition(self, leg_id, foot_local_position):
        """Use IK to compute the motor angles, given the foot link's local position.

        Args:
          leg_id: The leg index.
          foot_local_position: The foot link's position in the base frame.

        Returns:
          A tuple. The position indices and the angles for all joints along the
          leg. The position indices is consistent with the joint orders as returned
          by GetMotorAngles API.
        """
        assert len(self._foot_link_ids) == constants.NUM_LEGS
        # toe_id = self._foot_link_ids[leg_id]

        motors_per_leg = constants.MOTORS_PER_LEG
        joint_position_idxs = list(
            range(leg_id * motors_per_leg,
                  leg_id * motors_per_leg + motors_per_leg))

        joint_angles = kinematics.foot_position_in_hip_frame_to_joint_angle(
            foot_local_position - constants.HIP_OFFSETS[leg_id],
            l_hip_sign=(-1)**(leg_id + 1))

        # Joint offset is necessary for Laikago.
        joint_angles = np.multiply(
            np.asarray(joint_angles) -
            np.asarray(self._motor_offset)[joint_position_idxs],
            self._motor_direction[joint_position_idxs])

        # Return the joing index (the same as when calling GetMotorAngles) as well
        # as the angles.
        return joint_position_idxs, joint_angles.tolist()

    def GetFootPositionsInBaseFrame(self):
        """Get the robot's foot position in the base frame."""
        motor_angles = self.GetMotorAngles()
        return kinematics.foot_positions_in_base_frame(motor_angles)

    def ComputeJacobian(self, leg_id):
        """Compute the Jacobian for a given leg."""
        # Does not work for Minitaur which has the four bar mechanism for now.
        motor_angles = self.GetMotorAngles()[leg_id * 3:(leg_id + 1) * 3]
        return kinematics.analytical_leg_jacobian(motor_angles, leg_id)

    def GetAllSensors(self):
        """get all sensors associated with this robot.

        Returns:
        sensors: a list of all sensors.
        """
        return self._sensors

    def GetSensor(self, name):
        """get the first sensor with the given name.

        This function return None if a sensor with the given name does not exist.

        Args:
        name: the name of the sensor we are looking

        Returns:
        sensor: a sensor with the given name. None if not exists.
        """
        for s in self._sensors:
            if s.get_name() == name:
                return s
        return None

    def SetAllSensors(self, sensors):
        """set all sensors to this robot and move the ownership to this robot.

        Args:
        sensors: a list of sensors to this robot.
        """
        for s in sensors:
            s.set_robot(self)
        self._sensors = sensors

    def ReceiveObservation(self):
        """Receive the observation from sensors.

        This function is called once per step. The observations are only updated
        when this function is called.
        """
        self._joint_states = self._pybullet_client.getJointStates(self.quadruped, self._motor_id_list)
        self._base_position, orientation = self._pybullet_client.getBasePositionAndOrientation(self.quadruped)
        # Computes the relative orientation relative to the robot's
        # initial_orientation.
        _, self._base_orientation = self._pybullet_client.multiplyTransforms(
            positionA=[0, 0, 0],
            orientationA=orientation,
            positionB=[0, 0, 0],
            orientationB=self._init_orientation_inv)
        self._observation_history.appendleft(self.GetTrueObservation())
        self._control_observation = self._GetControlObservation()
        self.last_state_time = self._state_action_counter * self.time_step

    def GetTrueObservation(self):
        observation = []
        observation.extend(self.GetTrueMotorAngles())
        observation.extend(self.GetTrueMotorVelocities())
        observation.extend(self.GetTrueMotorTorques())
        observation.extend(self.GetTrueBaseOrientation())
        observation.extend(self.GetTrueBaseRollPitchYawRate())
        return observation

    def GetTrueMotorAngles(self):
        """Gets the motor angles at the current moment, mapped to [-pi, pi].

        Returns:
        Motor angles, mapped to [-pi, pi].
        """
        motor_angles = [state[0] for state in self._joint_states]
        motor_angles = np.multiply(np.asarray(motor_angles) - np.asarray(self._motor_offset), self._motor_direction)
        return motor_angles

    def GetMotorAngles(self):
        """Gets the motor angles.

        This function mimicks the noisy sensor reading and adds latency. The motor
        angles that are delayed, noise polluted, and mapped to [-pi, pi].

        Returns:
        Motor angles polluted by noise and latency, mapped to [-pi, pi].
        """
        motor_angles = self._AddSensorNoise(
            np.array(self._control_observation[0:constants.NUM_MOTORS]),
            self._observation_noise_stdev[0])
        return MapToMinusPiToPi(motor_angles)

    def GetTrueMotorVelocities(self):
        """Get the velocity of all eight motors.

        Returns:
        Velocities of all eight motors.
        """
        motor_velocities = [state[1] for state in self._joint_states]
        motor_velocities = np.multiply(motor_velocities, self._motor_direction)
        return motor_velocities

    def GetMotorVelocities(self):
        """Get the velocity of all eight motors.

        This function mimicks the noisy sensor reading and adds latency.
        Returns:
        Velocities of all eight motors polluted by noise and latency.
        """
        return self._AddSensorNoise(
            np.array(self._control_observation[constants.NUM_MOTORS:2 * constants.NUM_MOTORS]),
            self._observation_noise_stdev[1])

    def GetTrueMotorTorques(self):
        """Get the amount of torque the motors are exerting.

        Returns:
        Motor torques of all eight motors.
        """
        return self._observed_motor_torques

    def GetMotorTorques(self):
        """Get the amount of torque the motors are exerting.

        This function mimicks the noisy sensor reading and adds latency.
        Returns:
        Motor torques of all eight motors polluted by noise and latency.
        """
        return self._AddSensorNoise(
            np.array(self._control_observation[2 * constants.NUM_MOTORS:3 * constants.NUM_MOTORS]),
            self._observation_noise_stdev[2])

    def GetTrueBaseOrientation(self):
        """Get the orientation of minitaur's base, represented as quaternion.

        Returns:
        The orientation of minitaur's base.
        """
        return self._base_orientation

    def GetBaseOrientation(self):
        """Get the orientation of minitaur's base, represented as quaternion.

        This function mimicks the noisy sensor reading and adds latency.
        Returns:
        The orientation of minitaur's base polluted by noise and latency.
        """
        return self._pybullet_client.getQuaternionFromEuler(
            self.GetBaseRollPitchYaw())

    def GetBasePosition(self):
        """Get the position of minitaur's base.

        Returns:
        The position of minitaur's base.
        """
        return self._base_position

    def GetBaseVelocity(self):
        """Get the linear velocity of minitaur's base.

        Returns:
        The velocity of minitaur's base.
        """
        velocity, _ = self._pybullet_client.getBaseVelocity(self.quadruped)
        return velocity

    def GetTrueBaseRollPitchYaw(self):
        """Get minitaur's base orientation in euler angle in the world frame.

        Returns:
        A tuple (roll, pitch, yaw) of the base in world frame.
        """
        orientation = self.GetTrueBaseOrientation()
        roll_pitch_yaw = self._pybullet_client.getEulerFromQuaternion(orientation)
        return np.asarray(roll_pitch_yaw)

    def GetBaseRollPitchYaw(self):
        """Get minitaur's base orientation in euler angle in the world frame.

        This function mimicks the noisy sensor reading and adds latency.
        Returns:
        A tuple (roll, pitch, yaw) of the base in world frame polluted by noise
        and latency.
        """
        delayed_orientation = np.array(self._control_observation[3 * constants.NUM_MOTORS:3 * constants.NUM_MOTORS + 4])
        delayed_roll_pitch_yaw = self._pybullet_client.getEulerFromQuaternion(delayed_orientation)
        roll_pitch_yaw = self._AddSensorNoise(np.array(delayed_roll_pitch_yaw), self._observation_noise_stdev[3])
        return roll_pitch_yaw


    def GetBaseRollPitchYawRate(self):
        """Get the rate of orientation change of the minitaur's base in euler angle.

        This function mimicks the noisy sensor reading and adds latency.
        Returns:
        rate of (roll, pitch, yaw) change of the minitaur's base polluted by noise
        and latency.
        """
        return self._AddSensorNoise(
            np.array(self._control_observation[3 * constants.NUM_MOTORS + 4:3 * constants.NUM_MOTORS + 7]),
            self._observation_noise_stdev[4])

    def GetTrueBaseRollPitchYawRate(self):
        """Get the rate of orientation change of the minitaur's base in euler angle.

        Returns:
        rate of (roll, pitch, yaw) change of the minitaur's base.
        """
        angular_velocity = self._pybullet_client.getBaseVelocity(self.quadruped)[1]
        orientation = self.GetTrueBaseOrientation()
        return self.TransformAngularVelocityToLocalFrame(angular_velocity, orientation)

    def TransformAngularVelocityToLocalFrame(self, angular_velocity, orientation):
        """Transform the angular velocity from world frame to robot's frame.

        Args:
        angular_velocity: Angular velocity of the robot in world frame.
        orientation: Orientation of the robot represented as a quaternion.

        Returns:
        angular velocity of based on the given orientation.
        """
        # Treat angular velocity as a position vector, then transform based on the
        # orientation given by dividing (or multiplying with inverse).
        # Get inverse quaternion assuming the vector is at 0,0,0 origin.
        _, orientation_inversed = self._pybullet_client.invertTransform([0, 0, 0], orientation)
        # Transform the angular_velocity at neutral orientation using a neutral
        # translation and reverse of the given orientation.
        relative_velocity, _ = self._pybullet_client.multiplyTransforms(
            [0, 0, 0], orientation_inversed, angular_velocity,
            self._pybullet_client.getQuaternionFromEuler([0, 0, 0]))
        return np.asarray(relative_velocity)

    def _AddSensorNoise(self, sensor_values, noise_stdev):
        if noise_stdev <= 0:
            return sensor_values
        observation = sensor_values + np.random.normal(scale=noise_stdev, size=sensor_values.shape)
        return observation

    def _GetControlObservation(self):
        control_delayed_observation = self._GetDelayedObservation(self._control_latency)
        return control_delayed_observation

    def _GetDelayedObservation(self, latency):
        """Get observation that is delayed by the amount specified in latency.

        Args:
        latency: The latency (in seconds) of the delayed observation.

        Returns:
        observation: The observation which was actually latency seconds ago.
        """
        if latency <= 0 or len(self._observation_history) == 1:
            observation = self._observation_history[0]
        else:
            n_steps_ago = int(latency / self.time_step)
            if n_steps_ago + 1 >= len(self._observation_history):
                return self._observation_history[-1]
            remaining_latency = latency - n_steps_ago * self.time_step
            blend_alpha = remaining_latency / self.time_step
            observation = (
                (1.0 - blend_alpha) *
                np.array(self._observation_history[n_steps_ago]) +
                blend_alpha * np.array(self._observation_history[n_steps_ago + 1]))
        return observation

    def GetFootLinkIDs(self):
        """Get list of IDs for all foot links."""
        return self._foot_link_ids

    def GetFootContacts(self):
        """Get minitaur's foot contact situation with the ground.

        Returns:
            A list of 4 booleans. The ith boolean is True if leg i is in contact with
            ground.
        """
        all_contacts = self._pybullet_client.getContactPoints(bodyA=self.quadruped)

        contacts = [False, False, False, False]
        for contact in all_contacts:
            # Ignore self contacts
            if contact[constants._BODY_B_FIELD_NUMBER] == self.quadruped:
                continue
            try:
                toe_link_index = self._foot_link_ids.index(contact[constants._LINK_A_FIELD_NUMBER])
                contacts[toe_link_index] = True
            except ValueError:
                continue

        return contacts

    @property
    def is_safe(self):
        return self._is_safe
