import numpy as np

class WalkAlongX(object):
    """Task to walk along a straight line (x-axis)"""
    def __init__(self,
                forward_reward_cap: float = float("inf"),
                distance_weight: float = 1.0,
                # energy_weight=0.0005,
                shake_weight: float = 0.005,
                drift_weight: float = 2.0,
                action_cost_weight: float = 0.5, 
                # deviation_weight: float = 1,
                enable_roll_limit: bool = True,
                healthy_roll_limit: float = np.pi * 3 / 4, 
                enable_z_limit: bool = False,
                healthy_z_limit: float = 0.2,
                healthy_reward=1.0,
                ):
        """Initializes the task."""

        self._forward_reward_cap = forward_reward_cap
        self._action_cost_weight = action_cost_weight
        self._distance_weight = distance_weight
        self._shake_weight = shake_weight
        self._drift_weight = drift_weight
        self.enable_roll_limit = enable_roll_limit
        self.healthy_roll_limit = healthy_roll_limit
        self.enable_z_limit = enable_z_limit
        self.healthy_z_limit = healthy_z_limit
        self.healthy_reward = healthy_reward

        self._current_base_pos = np.zeros(3)
        # self._last_base_pos = np.zeros(3)
        # self._cumulative_displacement = 0

    def __call__(self, env):
        return self.reward(env)

    def reset(self, env):
        """Resets the internal state of the task."""
        self._env = env
        
        self._current_base_pos = env.robot.GetBasePosition()
        # self._last_base_pos = self._current_base_pos
        
        self._current_base_vel = env.robot.GetBaseVelocity()
        # self._alive_time_reward = 0
        # self._cumulative_displacement = 0
        self._last_action = env.last_action
        
        self._current_base_ori_euler = env.robot.GetBaseRollPitchYaw()
        _current_base_ori_quat = env.robot.GetBaseOrientation()
        rot_matrix = env.pybullet_client.getMatrixFromQuaternion(_current_base_ori_quat)
        self._local_up_vec = rot_matrix[6:]
        
    def update(self, env):
        """Updates the internal state of the task.
        Evoked after call to a1.A1.Step(), ie after action takes effect in simulation
        """
        # self.last_base_pos = self._current_base_pos
        self._current_base_pos = env.robot.GetBasePosition()

        self._current_base_vel = env.robot.GetBaseVelocity()
        # self._alive_time_reward = env.get_time_since_reset()
        self._last_action = env.last_action
        
        self._current_base_ori_euler = env.robot.GetBaseRollPitchYaw()
        _current_base_ori_quat = env.robot.GetBaseOrientation()
        rot_matrix = env.pybullet_client.getMatrixFromQuaternion(_current_base_ori_quat)
        self._local_up_vec = rot_matrix[6:]
        # self._cumulative_displacement = 0.5 * self._cumulative_displacement + \
        #     self._current_base_pos[0] - self._last_base_pos[0]

    def done(self, env):
        """Checks if the episode is over.

            If the robot base becomes unstable (based on orientation), the episode
            terminates early.
        """
        del env
        return not self.is_healthy

    def reward(self, env):
        """Get the reward without side effects."""
        del env
        # x_cum_disp_reward = self._cumulative_displacement
        x_velocity_reward = self._current_base_vel[0]

        action_cost = self._action_cost_weight * np.linalg.norm(self._last_action)
        # the far the better..
        forward_reward = self._current_base_pos[0]
        # Cap the forward reward if a cap is set.
        forward_reward = min(forward_reward, self._forward_reward_cap)
        # Penalty for sideways translation.
        drift_reward = -abs(self._current_base_pos[1])
        # Penalty for sideways rotation of the body.
        shake_reward = -abs(np.dot(np.asarray([1, 1, 0]), np.asarray(self._local_up_vec)))
        # # Penalty for Energy consumption
        # energy_reward = -np.abs(np.dot(env.robot.GetMotorTorques(), env.robot.GetMotorVelocities())) * env.sim_time_step
        # objectives = [forward_reward, energy_reward, drift_reward, shake_reward]
        # weighted_objectives = [o * w for o, w in zip(objectives, self._objective_weights)]
        reward = forward_reward * self._distance_weight + \
                    drift_reward * self._drift_weight + \
                    shake_reward * self._shake_weight + \
                    action_cost * self._action_cost_weight * \
                    x_velocity_reward

        if self.is_healthy:
            reward += self.healthy_reward
        return reward 
    
    @property
    def is_healthy(self):
        # Check for counterclockwise rotation along x-axis (in radians)
        if self.enable_roll_limit and (
            np.any(self._current_base_ori_euler < -self.healthy_roll_limit) or \
            np.any(self._current_base_ori_euler > self.healthy_roll_limit)
            ):
            return False
        # Isuue - needs to account for heightfield data
        if self.enable_z_limit and self._current_base_pos[2] < self.healthy_z_limit:
            return False
        return True

