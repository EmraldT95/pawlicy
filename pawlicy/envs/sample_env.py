import time
import gym
import numpy as np
import pybullet as p
import pybullet_data as pbd
from pybullet_utils.bullet_client import BulletClient

from pawlicy.robots.a1 import A1_simple
from pawlicy.envs.terrains import TerrainRandomizer, TerrainConstants
from pawlicy.tasks import DefaultTask

_NUM_SIMULATION_ITERATION_STEPS=300

class SampleEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 66}

    def __init__(self,
                enable_rendering=False,
                num_action_repeat=33,
                sim_time_step_s=0.03,
                randomise_terrain=False,
                motor_control_mode="Position",
                task:DefaultTask=DefaultTask()):
        """Initializes the locomotion gym environment.

		Args:
            enable_rendering: Whether to run pybullet in GUI mode or not
            num_action_repeat: The number of simulation steps that the same action is repeated.
            sim_time_step_s: The simulation time step in PyBullet. By default, the simulation
                step is 0.001s, which is a good trade-off between simulation speed and
                accuracy.
            randomise_terrain: Whether to randomize the terrains or not
            motor_control_mode: The mode in which the robot will operate. This will determine
                the action space.
			task: A callable function/class to calculate the reward and termination
				condition. Takes the gym env as the argument when calling.

		Raises:
			ValueError: If the num_action_repeat is less than 1.

		"""
        self._world_dict = {} # A dictionary containing the objects in the world other than the robot.
        self._task = task
        self._is_render = enable_rendering
        self._num_action_repeat = num_action_repeat
        self._sim_time_step = sim_time_step_s
        self._env_time_step = self._sim_time_step * self._num_action_repeat
        self._randomise_terrain = randomise_terrain
        self._motor_control_mode = motor_control_mode
        self._last_frame_time = 0.0 # The wall-clock time at which the last frame is rendered.

        # Configure PyBullet
        if self._is_render:
            self._pybullet_client = BulletClient(connection_mode=p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, True)
        else:
            self._pybullet_client = BulletClient()
        self._pybullet_client.setAdditionalSearchPath(pbd.getDataPath())
        self._pybullet_client.setTimeStep(self._sim_time_step)
        self._pybullet_client.setGravity(0, 0, -100)

        self.reset(True) # Hard reset initially to load the robot URDF file


    def reset(self, hard_reset=False, initial_motor_angles=None, reset_duration=0.0):
        """Resets the robot's position in the world or rebuild the sim world.

		The simulation world will be rebuilt if self._hard_reset is True.

		Args:
			initial_motor_angles: A list of Floats. The desired joint angles after
			    reset. If None, the robot will use its built-in value.
			reset_duration: Float. The time (in seconds) needed to rotate all motors
			    to the desired initial values.

		Returns:
			A numpy array contains the initial observation after reset.
		"""
        if self._is_render:
            self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, 0)

        # Clear the simulation world and reset the robot interface.
        if hard_reset:
            self._pybullet_client.resetSimulation()

            # Randomising terrains, if needed
            terrain_id = -1
            if self._randomise_terrain:
                terrain_id, terrain_type = TerrainRandomizer(self._pybullet_client).randomize()
            else:
                terrain_id = self._pybullet_client.loadURDF("plane_implicit.urdf")
                terrain_type = "plane"

            self._world_dict = { 
                "terrain_id": terrain_id,
                "terrain_id": terrain_id
            }

            # Build the robot
            self._robot = A1_simple(
                    pybullet_client=self._pybullet_client,
                    time_step=self._sim_time_step,
                    action_repeat=self._num_action_repeat,
                    enable_action_interpolation=True,
                    motor_control_mode=self._motor_control_mode,
                    init_position=TerrainConstants.ROBOT_INIT_POSITION[terrain_type])

            # Create the action space and observation space
            self.action_space = self._build_action_space()
            self.observation_space = self._build_observation_space()
        else:
            # Reset the robot.
            self._robot.Reset(reload_urdf=False, default_motor_angles=initial_motor_angles, reset_time=reset_duration)

        self._env_step_counter = 0
        if self._is_render:
            self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, 1)
        self._pybullet_client.resetDebugVisualizerCamera(1.0, 0, -30, [0, 0, 0])
        self._last_action = np.zeros(self.action_space.shape)
        self._task.reset(self) # Reset the state of the task as well

        return self._get_observation()


    def step(self, action):
        """Step forward the simulation, given the action.

        Args:
            action: Can be a list of desired motor angles for all motors when the
                robot is in position control mode; A list of desired motor torques. Or a
                list of velocities for velocity control mode. The
                action must be compatible with the robot's motor control mode.

        Returns:
            observations: The observation based on current action
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
            # time_spent = time.time() - self._last_frame_time
            # self._last_frame_time = time.time()
            # time_to_sleep = self._env_time_step - time_spent
            # if time_to_sleep > 0:
            #     time.sleep(time_to_sleep)
            base_pos = self._robot.GetBasePosition()

            # Also keep the previous orientation of the camera set by the user.
            [yaw, pitch, dist] = self._pybullet_client.getDebugVisualizerCamera()[8:11]
            self._pybullet_client.resetDebugVisualizerCamera(dist, yaw, pitch, base_pos)
            self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_SINGLE_STEP_RENDERING, 1)

        # Execute the action
        self._act(action)

        observation = self._get_observation()
        reward = self._task.reward(self)
        done = self._task.done(self)
        self._env_step_counter += 1
        return observation, reward, done, {}


    def render(self, mode='rgb_array'):
        """
        Renders the rgb view from the robots perspective.
        Currently tuned to get the view from the head of the robot
        """
        # Base information
        proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(fov=66, aspect=1, nearVal=0.01, farVal=100)
        base_pos = list(self._robot.GetBasePosition())
        base_ori = list(self._robot.GetTrueBaseOrientation())
        base_pos[2] = base_pos[2]+0.1

        # Rotate camera direction
        rot_mat = np.array(self._pybullet_client.getMatrixFromQuaternion(base_ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0]) # Target position is always the x-axis. This decides the diirection of the camera
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1])) # Which direction is considered "up"
        view_matrix = self._pybullet_client.computeViewMatrix(base_pos, base_pos + camera_vec, up_vec)

        (_, _, px, _, _) = self._pybullet_client.getCameraImage(
            width=480,
            height=360,
            renderer=self._pybullet_client.ER_BULLET_HARDWARE_OPENGL,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array


    def close(self):
        """Terminates the simulation"""
        self._robot.Terminate()


    def _build_action_space(self):
        """Defines the action space of the gym environment"""
        # All limits defined according to urdf file
        joint_limits = self._robot.GetJointLimits()
        # Controls the torque applied at each motor
        if self._motor_control_mode == "Torque":
            high = joint_limits["torque"]
            action_space = gym.spaces.Box(-high, high, dtype=np.float32)
        # Controls the angles (in radans) of the joints
        elif self._motor_control_mode == "Position":
            action_space = gym.spaces.Box(joint_limits["lower"], joint_limits["upper"], dtype=np.float32)
        # Controls the velocity at which motors rotate
        elif self._motor_control_mode == "Velocity":
            high = joint_limits["velocity"]
            action_space = gym.spaces.Box(-high, high, dtype=np.float32)
        else:
            raise ValueError
        return action_space


    def _build_observation_space(self):
        """Defines the observation space of the gym environment"""
        high = []
        low = []

        trunk_pos_limit_high = [100] * 3
        trunk_pos_limit_low = [-100] * 3
        high.extend(trunk_pos_limit_high)
        low.extend(trunk_pos_limit_low)

        trunk_ori_limit_high = [1] * 4
        trunk_ori_limit_low = [-1] * 4
        high.extend(trunk_ori_limit_high)
        low.extend(trunk_ori_limit_low)

        motor_angle_limit_high  = [np.pi] * 12
        motor_angle_limit_low  = [-np.pi] * 12
        high.extend(motor_angle_limit_high)
        low.extend(motor_angle_limit_low)

        trunk_lin_vel_high = [10] * 3
        trunk_lin_vel_low = [-10] * 3
        high.extend(trunk_lin_vel_high)
        low.extend(trunk_lin_vel_low)

        trunk_ang_vel_high = [200 * np.pi] * 3
        trunk_ang_vel_low = [-200 * np.pi] * 3
        high.extend(trunk_ang_vel_high)
        low.extend(trunk_ang_vel_low)

        link_ang_vel_high = [200] * 3 * 12
        link_ang_vel_low = [-200] * 3 * 12
        high.extend(link_ang_vel_high)
        low.extend(link_ang_vel_low)

        high=np.array(high)
        low=np.array(low)
        observation_space = gym.spaces.Box(low, high, dtype=np.float64)
        return observation_space


    def _act(self, action):
        """Executes the action in the robot and also updates the task state"""
        self._robot.TakeAction(action)
        self._task.update(self)


    def _get_observation(self):
        """Get observation based on the information about the robots current state"""
        return self._robot.GetObservation()


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
    def last_action(self):
        return self._last_action

    @property
    def last_base_position(self):
        return self._last_base_position

    @property
    def env_step_counter(self):
        return self._env_step_counter

    @property
    def world_dict(self):
        return self._world_dict.copy()

    @world_dict.setter
    def world_dict(self, new_dict):
        self._world_dict = new_dict.copy()
