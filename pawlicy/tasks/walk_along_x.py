import numpy as np

class WalkAlongX(object):
    """Task to walk along a straight line (x-axis)"""
    def __init__(self,
                action_cost_weight: float = 0.5, 
                deviation_weight: float = 1,
                enable_roll_limit: bool = True,
                healthy_roll_limit: float = np.pi * 3 / 4, 
                enable_z_limit: bool = False,
                healthy_z_limit: float = 0.2,
                healthy_reward=1.0,
                ):
        """Initializes the task."""

        self._action_cost_weight = action_cost_weight
        self._deviation_weight = deviation_weight
        self.enable_roll_limit = enable_roll_limit
        self.healthy_roll_limit = healthy_roll_limit
        self.enable_z_limit = enable_z_limit
        self.healthy_z_limit = healthy_z_limit
        self.healthy_reward = healthy_reward

        self.current_base_pos = np.zeros(3)

    def __call__(self, env):
        return self.reward(env)

    def reset(self, env):
        """Resets the internal state of the task."""
        self._env = env

        self._current_base_pos = env.robot.GetBasePosition()
        self._current_base_vel = env.robot.GetBaseVelocity()
        self._alive_time_reward = 0
        self._last_action = env.last_action
        
        self._current_base_ori_euler = env.robot.GetTrueBaseRollPitchYaw()

    def update(self, env):
        """Updates the internal state of the task.
        Evoked after call to a1.A1.Step(), ie after action takes effect in simulation
        """
        self._current_base_pos = env.robot.GetBasePosition()
        self._current_base_vel = env.robot.GetBaseVelocity()
        self._alive_time_reward = env.get_time_since_reset()
        self._last_action = env.last_action
        
        self._current_base_ori_euler = env.robot.GetTrueBaseRollPitchYaw()   

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
        x_velocity_reward = self._current_base_vel[0]

        action_cost = self._action_cost_weight * np.linalg.norm(self._last_action)
        y_deviation_cost = self._deviation_weight * self.current_base_pos[1] ** 2 

        total = x_velocity_reward + self._alive_time_reward \
                - action_cost - y_deviation_cost
        
        if self.is_healthy:
            total += self.healthy_reward
        return total
    
    @property
    def is_healthy(self):
        # Check for counterclockwise rotation along x-axis (in radians)
        if self.enable_roll_limit and (self._current_base_ori_euler[0] < -self.healthy_roll_limit or \
            self._current_base_ori_euler[0] > self.healthy_roll_limit):
            return False
        # Isuue - needs to account for heightfield data
        if self.enable_z_limit and self.current_base_pos[2] < self.healthy_z_limit:
            return False
        return True
