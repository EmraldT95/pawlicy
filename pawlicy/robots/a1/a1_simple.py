from pawlicy.robots.a1 import constants
from pawlicy.utils import env_utils

import copy
import collections
import re
import math
import pybullet as p
import numpy as np
from typing import Union

INIT_ORIENTATION = [0, 0, 0, 1]

HIP_NAME_PATTERN = re.compile(r"\w+_hip_\w+")
UPPER_NAME_PATTERN = re.compile(r"\w+_upper_\w+")
LOWER_NAME_PATTERN = re.compile(r"\w+_lower_\w+")
TOE_NAME_PATTERN = re.compile(r"\w+_toe\d*")
IMU_NAME_PATTERN = re.compile(r"imu\d*")


class A1(object):
    link_name2id = {
        "trunk" : -1,
        "imu_link" : 0,
        "FR_hip" : 1,
        "FR_upper_shoulder" : 2,
        "FR_upper" : 3,
        "FR_lower" : 4,
        "FR_toe" : 5,
        "FL_hip" : 6,
        "FL_upper_shoulder" : 7,
        "FL_upper" : 8,
        "FL_lower" : 9,
        "FL_toe" : 10,
        "RR_hip" : 11,
        "RR_upper_shoulder" : 12,
        "RR_upper" : 13,
        "RR_lower" : 14,
        "RR_toe" : 15,
        "RL_hip" : 16,
        "RL_upper_shoulder" : 17,
        "RL_upper" : 18,
        "RL_lower" : 19,
        "RL_toe" : 20,
    }

    def __init__(
        self,
        pybullet_client,
        enable_clip_motor_commands=False,
        time_step=0.001,
        action_repeat=10,
        enable_action_interpolation=True,
        reset_time=1,
        self_collision_enabled=False,
        motor_torque_limits=None,
        motor_control_mode="Position",
        init_position=[0, 0, 1]
    ):
        self._pybullet_client = pybullet_client
        self._time_step = time_step
        self._action_repeat = action_repeat
        self._reset_time = reset_time
        self._self_collision_enabled = self_collision_enabled
        self._motor_torque_limits = motor_torque_limits
        self._motor_control_mode = motor_control_mode
        self._init_position = init_position # This will be overwritten based on the terrain loaded
        self._init_orientation = self._pybullet_client.getQuaternionFromEuler([0., 0., 0.])
        self._joint_name_to_id = {}

        # Load the URDF file of the robot
        self.quadruped = self._LoadRobotURDF()

        # Get some information about the joints of the robot
        self._num_joints = self._pybullet_client.getNumJoints(self.quadruped)
        self._GetJointsInfo()
        self._GetLinksInfo()

        # Reset the robot initially
        self.Reset()

        # self._log_general()
        self.disable_motor()


    def Reset(self, reset_time=0.3):
        """Reset the robot to its initial states.

        Args:
        reset_time: The duration (in seconds) to hold the default motor angles. If
            reset_time <= 0 or in torque control mode, the phase of holding the
            default pose is skipped.
        """
        self._step_counter = 0
        self._state_action_counter = 0
        self._last_action = None
        self._pybullet_client.resetBasePositionAndOrientation(self.quadruped, self._init_position, self._init_orientation)
        self._pybullet_client.resetBaseVelocity(self.quadruped, [0, 0, 0], [0, 0, 0])
        # Inverse of global base orientation
        _, self._init_base_orientation_inv = self._pybullet_client.invertTransform(position=[0, 0, 0], orientation=self._init_orientation)
        # Reset the pose of the Robot
        self.ResetPose()
        # Gets the robot joint and links states
        self.UpdateStates()


    def ResetPose(self):
        """
        Resets the pose of the robot to its initial pose
        """
        # for name in self._joint_name_to_id:
        #     joint_id = self._joint_name_to_id[name]
        #     # Setting force to 0 disables the default velocity motor from pybullet
        #     self._pybullet_client.setJointMotorControl2(
        #         bodyIndex=self.quadruped,
        #         jointIndex=joint_id,
        #         controlMode=self._pybullet_client.VELOCITY_CONTROL,
        #         targetVelocity=0,
        #         force=0)

        # Set the angle(in radians) for each joint
        for name, i in zip(constants.JOINT_NAMES, range(len(constants.JOINT_NAMES))):
            self._pybullet_client.resetJointState(self.quadruped,
                                                  self._joint_name_to_id[name],
                                                  constants.INIT_MOTOR_ANGLES[i],
                                                  targetVelocity=0)


    def TakeAction(self, action):
        """Executes the action on the robot based on the motor control mode"""
        self._last_action = action
        if self._motor_control_mode == "Torque":
            controlMode = self._pybullet_client.TORQUE_CONTROL
        elif self._motor_control_mode == "Position":
            controlMode = self._pybullet_client.POSITION_CONTROL
        elif self._motor_control_mode == "Velocity":
            controlMode = self._pybullet_client.VELOCITY_CONTROL
        else:
            raise ValueError

        # Interpolate the action to make the action changes smooth
        # if self._last_action is not None:
        #     for i in range(self._action_repeat):
        #         lerp = float(i + 1) / self._action_repeat
        #         action = self._last_action + lerp * (action - self._last_action)
        # else:
        #     action = action
        # Apply the action
        self._pybullet_client.setJointMotorControlArray(self.quadruped, 
                                    self._joint_id_list,
                                    controlMode,
                                    forces=action)
        self._pybullet_client.stepSimulation()
        self._step_counter += 1
        self.UpdateStates()


    def GetObservation(self):
        """
        Gets information about the robot's pose, orientation, velocity, etc. 
        as an observation
        """
        observation = []
        trunk_pos = self.GetBasePosition() # tuple(3)
        observation.extend(trunk_pos)
        
        trunk_ori = self.GetTrueBaseOrientation() # tuple(4), quat
        observation.extend(trunk_ori)
        
        
        joint_angles = self.GetTrueMotorAngles() # list(self.num_joints)
        observation.extend(joint_angles)

        trunk_vel = self.GetBaseVelocity() # tuple(3)
        trunk_ang_vel = self.GetTrueBaseRollPitchYawRate() # tuple(3)
        observation.extend(trunk_vel)
        observation.extend(trunk_ang_vel)

        link_ang_vels = self.GetRawLinkAngularVelocity() # tuple(3)
        observation.extend(link_ang_vels)
        
        return np.asarray(observation)


    def _LoadRobotURDF(self):
        """
        Loads the URDF file of the robot from pybullet_data
        """
        if self._self_collision_enabled:
            return self._pybullet_client.loadURDF(constants.URDF_FILEPATH,
                                                self.init_position,
                                                INIT_ORIENTATION,
                                                flags=self._pybullet_client.URDF_USE_SELF_COLLISION)
        else:
            return self._pybullet_client.loadURDF(constants.URDF_FILEPATH,
                                                self.init_position,
                                                INIT_ORIENTATION)

                
    def _GetJointsInfo(self):
        """
        This function does 3 things:
        1) Creates a dictionary maps the joint name to its ID.
        2) Gets information about the limits applied on each
            joint based on the URDF file.
        3) Changes how the joint reacts with the environment

        Raises:
          ValueError: Unknown category of the joint name.
        """
        self._num_joints = self._pybullet_client.getNumJoints(self.quadruped)
        self._hip_link_ids = []
        self._upper_link_ids = []
        self._lower_link_ids = []
        self._foot_link_ids = []
        self._imu_link_ids = []
        self._joint_lower_limits = []
        self._joint_upper_limits = []
        self._joint_max_force = []
        self._joint_max_velocity = []
        self._joint_id_list = []

        for i in range(self._num_joints):
            joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
            joint_id = joint_info[0]
            joint_name = joint_info[1].decode("UTF-8")
            self._joint_name_to_id[joint_name] = joint_id

            # Storing the ID each of each in a seperate array - might come handy later
            if HIP_NAME_PATTERN.match(joint_name):
                self._hip_link_ids.append(joint_id)
            elif UPPER_NAME_PATTERN.match(joint_name):
                self._upper_link_ids.append(joint_id)
            elif LOWER_NAME_PATTERN.match(joint_name):
                self._lower_link_ids.append(joint_id)
            elif TOE_NAME_PATTERN.match(joint_name):
                self._foot_link_ids.append(joint_id)
            elif IMU_NAME_PATTERN.match(joint_name):
                self._imu_link_ids.append(joint_id)
            else:
                raise ValueError("Unknown category of joint %s" % joint_name)
            
            # Setting up the dynamics of each joint relative to the body
            self._pybullet_client.changeDynamics(joint_id, -1, linearDamping=0, angularDamping=0)

            # Getting lower limit, upper limit, max force and max velocity of each joint for building the actions the environment
            if joint_name in constants.JOINT_NAMES:
                self._joint_lower_limits.append(joint_info[8])
                self._joint_upper_limits.append(joint_info[9])
                self._joint_max_force.append(joint_info[10])
                self._joint_max_velocity.append(joint_info[11])
                self._joint_id_list.append(joint_id)

        self._hip_link_ids.sort()
        self._upper_link_ids.sort()
        self._lower_link_ids.sort()
        self._foot_link_ids.sort()


    def _GetLinksInfo(self, only_mesh=True):
        """
        Gets information about the links
        """
        MESH_GEOMENTRY_TYPE = 5
        self._link_id_list = []
        shape_info = self._pybullet_client.getVisualShapeData(self.quadruped)

        if not only_mesh:
            self._link_id_list = list(range(len(shape_info)))
        else:
            for link in shape_info:
                if link[2] == MESH_GEOMENTRY_TYPE and link[1] != -1:
                    self._link_id_list.append(link[1])


    def UpdateStates(self):
        """Updates the variables holding link and joint state information"""
        self._link_states = self._pybullet_client.getLinkStates(self.quadruped, self._link_id_list, computeLinkVelocity=1)
        self._joint_states = self._pybullet_client.getJointStates(self.quadruped, self._joint_id_list)
        env_utils._log("Link States", self._link_states,header="link_states")
        env_utils._log("Joint States", self._joint_states, header="joint_states")


    # def _log_general(self):
    #     joint_infos = [self._pybullet_client.getJointInfo(self.quadruped, joint_number) 
    #                 for joint_number in range(self._pybullet_client.getNumJoints(self.quadruped))]
    #     env_utils._log("Joint Informations", joint_infos, header="joint_infos")

    #     shape_info = self._pybullet_client.getVisualShapeData(self.quadruped)
    #     env_utils._log("Visual Shape info", shape_info,header="shape_infos")
    
    def disable_motor(self):
        self._pybullet_client.setJointMotorControlArray(self.quadruped,
                                self._joint_id_list,
                                controlMode=self._pybullet_client.VELOCITY_CONTROL,
                                forces=np.zeros(len(self._joint_id_list)).tolist())

    def GetJointLimits(self):
        """Gets the joint limits (angle, torque and velocity) of the robot"""
        return {
            "lower": np.array(self._joint_lower_limits),
            "upper": np.array(self._joint_upper_limits),
            "torque": np.array(self._joint_max_force),
            "velocity": np.array(self._joint_max_velocity)
        }


    def GetBasePosition(self):
        """
        Get the position of quadruped's base.
            
        Returns:
            The position of quadruped's base.
        """
        base_position , _ = self._pybullet_client.getBasePositionAndOrientation(self.quadruped)
        return base_position


    def GetBaseVelocity(self):
        """Get the linear velocity of quadruped's base.
        Returns:
        The velocity of quadruped's base.
        """
        velocity = self._pybullet_client.getBaseVelocity(self.quadruped)[0]
        env_utils._log(desc="Base Linear Velocity", msg=velocity, is_tabular=False)
        return velocity


    def GetTrueBaseOrientation(self):
        """
        Get the orientation of quadruped's base, represented as quaternion. Computes the 
        relative orientation relative to the robot's initial_orientation.
        """
        _, base_orientation = self._pybullet_client.getBasePositionAndOrientation(self.quadruped)
        _, base_orientation = self._pybullet_client.multiplyTransforms(
            positionA=[0, 0, 0],
            orientationA=base_orientation,
            positionB=[0, 0, 0],
            orientationB=self._init_base_orientation_inv)
        return base_orientation



    def GetTrueBaseRollPitchYaw(self):
        """
        Get quadruped's base orientation in euler angle in the world frame.
        """
        orientation = self.GetTrueBaseOrientation()
        roll_pitch_yaw = self._pybullet_client.getEulerFromQuaternion(orientation)
        return np.asarray(roll_pitch_yaw)


    def GetTrueBaseRollPitchYawRate(self):
        """
        Get the rate of orientation change of the quadruped's base in euler angle.
        
        Returns:
            rate of (roll, pitch, yaw) change of the quadruped's base.
        """
        angular_velocity = self._pybullet_client.getBaseVelocity(self.quadruped)[1]
        orientation = self.GetTrueBaseOrientation()
        angular_velocity_local = self.TransformAngularVelocityToLocalFrame(angular_velocity, orientation)
        env_utils._log("Base Angular Velocity", angular_velocity_local, is_tabular=False)
        return angular_velocity_local


    def TransformAngularVelocityToLocalFrame(self, angular_velocity, orientation):
        """
        Transform the angular velocity from world frame to robot's frame.
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


    def GetRawLinkAngularVelocity(self):
        """
        Gets the angular velocity
        """
        ang_velocities = []
        for link in self._link_states:
            ang_velocities.extend(link[-1])
        
        env_utils._log("Link Raw Angular Velocity", np.asarray(ang_velocities).reshape(12, 3), header="link_angular_velocities")
        return np.asarray(ang_velocities)


    def GetTrueMotorAngles(self):
        """Gets the motor angles at the current moment, mapped to [-pi, pi].
        Returns:
        Motor angles, mapped to [-pi, pi].
        """
        motor_angles = []
        for joint in self._joint_states:
            joint_angle = env_utils.MapToMinusPiToPi(joint[0]) # scalar, jointPosition
            motor_angles.append(joint_angle)
        return motor_angles


    def GetTrueMotorVelocities(self):
        """
        Get the velocity of all motors.
        
        Returns:
            Velocities of all motors.
        """
        motor_velocities = [state[1] for state in self._joint_states]
        return motor_velocities


    def GetFootLinkIDs(self):
        """Get list of IDs for all foot links."""
        return self._foot_link_ids


    def GetTimeSinceReset(self):
        """Get time since the last reset"""
        return self._step_counter * self._time_step


    @property
    def init_position(self):
        return self._init_position
    
    @init_position.setter
    def init_position(self, new_position):
        self._init_position = new_position
    
    @property
    def init_orientation(self):
        return self._init_orientation

    @property
    def init_orientation(self):
        return self._init_position

    @property
    def num_joints(self):
        return self._num_joints
        
    @property
    def pybullet_client(self):
        return self._pybullet_client

    @property
    def robot_id(self):
        return self.quadruped
