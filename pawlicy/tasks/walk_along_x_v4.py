import numpy as np

class WalkAlongX(object):
    """Task to walk along a straight line (x-axis)"""
    def __init__(self,
                #forward_reward_cap: float = float("inf"),
                velocity_weight: float = 10,
                distance_weight: float = 0.1,
                step_weight : float = 1,
                # energy_weight=0.0005,
                shake_weight: float = 0.005,
                drift_weight: float = 1,
                orientation_weight : float = 1,
                action_cost_weight: float = 0.02,
                # deviation_weight: float = 1,
                enable_roll_limit : bool = False,
                healthy_roll_limit : float = np.pi * 3/4,
                # roll_threshold: float = np.pi * 1/2,
                # pitch_threshold: float = 0.8,
                enable_z_limit: bool = True,
                healthy_z_limit: float = 0.15,
                # healthy_reward=1.0,
                ):
        """Initializes the task."""

        #self._forward_reward_cap = forward_reward_cap
        self._action_cost_weight = action_cost_weight
        self._velocity_weight = velocity_weight
        self._distance_weight = distance_weight
        self._shake_weight = shake_weight
        self._drift_weight = drift_weight
        self._step_weight = step_weight
        self._orientation_weight = orientation_weight
        # self._deviation_weight = deviation_weight
        #self.roll_threshold = roll_threshold
        self.enable_z_limit = enable_z_limit
        self.healthy_z_limit = healthy_z_limit
        # self.healthy_reward = healthy_reward
        #self.pitch_threshold = pitch_threshold
        self.enable_roll_limit = enable_roll_limit
        self.healthy_roll_limit = healthy_roll_limit

        self.step_counter = 0

        self._target_pos =  np.array([50, 0, 1])

    def __call__(self, env):
        return self.reward(env)

    def reset(self, env):
        """Resets the internal state of the task."""
        self._env = env
        self._init_base_pos = env.robot.GetBasePosition()
        self._current_base_pos = env.robot.GetBasePosition()
        self._last_base_pos = self._current_base_pos
        
        self._current_base_vel = env.robot.GetBaseVelocity()
        # self._alive_time_reward = 0
        # self._cumulative_displacement = 0
        self._last_action = env.robot.last_action
        
        #self._current_base_ori_euler = self.GetRawOrientation(env)
        self._current_base_ori_euler = env.robot.GetRawBaseRollPitchYaw()
        self._init_base_ori_euler = self._current_base_ori_euler
        #print(f"Initial Orienttaion {self._init_base_ori_euler}")
        self.step_counter = 0

        
    def update(self, env):
        """Updates the internal state of the task.
        Evoked after call to a1.A1.Step(), ie after action takes effect in simulation
        """
        self._last_base_pos = self._current_base_pos
        self._current_base_pos = env.robot.GetBasePosition()

        self._current_base_vel = env.robot.GetBaseVelocity()
        # self._alive_time_reward = env.get_time_since_reset()
        self._last_action = env.last_action
        
        #self._current_base_ori_euler = self.GetRawOrientation(env)
        self._current_base_ori_euler = env.robot.GetRawBaseRollPitchYaw()

        self.step_counter += self._step_weight

    def done(self, env):
        """Checks if the episode is over.

            If the robot base becomes unstable (based on orientation), the episode
            terminates early.
        """
        
        return not self.is_healthy


    def reward(self, env):

        # bug : -ve velocity along y is rewarded
        #velocity_reward = np.dot([1, -1, 0], self._current_base_vel)
        velocity_reward = self._velocity_weight * self._current_base_vel[0]
        # # forward_reward = self._distance_weight * (self._current_base_pos[0] - self._init_base_pos[0])
        # #displacement_reward = self._current_base_pos[0] - self._last_base_pos[0]

        # # y_velocity_reward = -abs(self._current_base_vel[1])
        action_reward = -self._action_cost_weight * np.linalg.norm(self._last_action) / 12
        
        
       
        orientation = env.robot.GetBaseOrientation()
        rot_matrix = env.robot._pybullet_client.getMatrixFromQuaternion(orientation)
        local_up_vec = rot_matrix[6:]
        shake_reward = - self._shake_weight * abs(np.dot(np.asarray([1, 1, 0]), np.asarray(local_up_vec)))
        # drift_reward =  - self._drift_weight * (self._current_base_pos[1])  ** 2
        # #distance_reward = - self._distance_weight * np.linalg.norm(self._target_pos - self._current_base_pos)
        # orientation_reward = - self._orientation_weight * np.linalg.norm(env.robot.GetTrueBaseRollPitchYaw() - self._init_base_ori_euler)
        reward = velocity_reward + shake_reward + self._step_weight #+ action_reward
            # x_velocity_reward + drift_reward + self.step_counter + distance_reward + orientation_reward
            #+ shake_reward # + y_velocity_reward + forward_reward + displacement_reward + action_reward \
                  #
        #print("Reward", reward)
        return reward

    @property
    def is_healthy(self):
        # Check for counterclockwise rotation along x-axis and y-axis (in radians)
        if self.enable_roll_limit:
            aug_vec = np.array([1, -1, 1])
            orientation_aug = self._current_base_ori_euler * aug_vec
            if np.any(orientation_aug < -self.healthy_roll_limit) or np.any(orientation_aug > self.healthy_roll_limit):
                return False

        # Isuue - needs to account for heightfield data
        if self.enable_z_limit and (self._current_base_pos[2] < self.healthy_z_limit):
            return False

        return True

    def is_fallen(self, env):
        """Decide whether Rex has fallen.
        If the up directions between the base and the world is larger (the dot
        product is smaller than 0.85) or the base is very low on the ground
        (the height is smaller than 0.13 meter), rex is considered fallen.
        Returns:
          Boolean value that indicates whether rex has fallen.
        """
        orientation = env.robot.GetBaseOrientation()
        rot_mat = env.robot._pybullet_client.getMatrixFromQuaternion(orientation)
        local_up = rot_mat[6:]
        return np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < 0.85