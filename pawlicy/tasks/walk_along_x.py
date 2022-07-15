import numpy as np
from pawlicy.envs.terrains import constants as terrain_constants 

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
                roll_threshold: float = np.pi * 0.3,
                pitch_threshold: float = 0.8,
                enable_z_limit: bool = True,
                healthy_z_limit: float = 0.05,
                healthy_reward=1.0,
                ):
        """Initializes the task."""

        self._forward_reward_cap = forward_reward_cap
        self._action_cost_weight = action_cost_weight
        self._distance_weight = distance_weight
        self._shake_weight = shake_weight
        self._drift_weight = drift_weight
        # self._deviation_weight = deviation_weight
        self.roll_threshold = roll_threshold
        self.enable_z_limit = enable_z_limit
        self.healthy_z_limit = healthy_z_limit
        self.healthy_reward = healthy_reward
        self.pitch_threshold = pitch_threshold

    def __call__(self, env):
        return self.reward(env)

    def reset(self, env):
        """Resets the internal state of the task."""
        self._env = env
        self._init_base_pos = env.robot.GetBasePosition()
        self._current_base_pos = env.robot.GetBasePosition()
        self._current_base_vel = env.robot.GetBaseVelocity()
        self._alive_time_reward = 0
        self._last_action = env.last_action
        
        self._init_base_ori_euler = env.robot.GetTrueBaseRollPitchYaw()
        self._current_base_ori_euler = env.robot.GetTrueBaseRollPitchYaw()

    def update(self, env):
        """Updates the internal state of the task.
        Evoked after call to a1.A1.Step(), ie after action takes effect in simulation
        """
        self.last_base_pos = self._current_base_pos
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
        return not self.is_healthy(env)

    def reward(self, env):
        """Get the reward without side effects."""
        # del env
        # # the far the better..
        x_velocity_reward = self._current_base_vel[0]

        
        # y_deviation_cost = self._deviation_weight * self.current_base_pos[1] ** 2 

        # total = x_velocity_reward + self._alive_time_reward \
        #         - action_cost - y_deviation_cost

        # the far the better..
        forward_reward = self._current_base_pos[0] - self._init_base_pos[0]
        # Cap the forward reward if a cap is set.
        forward_reward = min(forward_reward, self._forward_reward_cap)
        # Rewarded if moving forward, else give only 20% of actual reward
        forward_reward = forward_reward * (self._distance_weight if forward_reward > 0 else self._distance_weight * 0.2)

        # Cost of executing action
        action_cost = self._action_cost_weight * np.linalg.norm(self._last_action)
        action_cost = action_cost * self._action_cost_weight

        # Penalty for sideways translation.
        drift_reward = -abs(self._current_base_pos[1])
        drift_reward = drift_reward * self._drift_weight

        # # Penalty for sideways rotation of the body.
        # orientation = env.robot.GetBaseOrientation()
        # rot_matrix = env.pybullet_client.getMatrixFromQuaternion(orientation)
        # local_up_vec = rot_matrix[6:]
        # shake_reward = -abs(np.dot(np.asarray([1, 1, 0]), np.asarray(local_up_vec)))
        # # Penalty for Energy consumption
        # energy_reward = -np.abs(np.dot(env.robot.GetMotorTorques(), env.robot.GetMotorVelocities())) * env.sim_time_step
        # objectives = [forward_reward, energy_reward, drift_reward, shake_reward]
        # weighted_objectives = [o * w for o, w in zip(objectives, self._objective_weights)]
        reward = forward_reward + action_cost + drift_reward + x_velocity_reward
                    
                    # drift_reward * self._drift_weight + \
                    # shake_reward * self._shake_weight + \
        
        if self.is_healthy(env):
            reward += self.healthy_reward
        return reward 
    
    def is_healthy(self, env):
        # Check for counterclockwise rotation along x-axis (in radians)
        # if self.enable_roll_limit and (self._current_base_ori_euler[0] < -self.healthy_roll_limit or \
        #     self._current_base_ori_euler[0] > self.healthy_roll_limit):
        #     return False

        # Checking if robot is in contact with the ground
        foot_links = env.robot.GetFootLinkIDs()
        ground = env.get_ground()
        # Skip the first env step
        if env.env_step_counter > 0:
            robot_ground_contacts = env.pybullet_client.getContactPoints(bodyA=env.robot.quadruped, bodyB=ground["id"])
            for contact in robot_ground_contacts:
                # Only the toes of the robot should in contact with the ground
                if contact[3] not in foot_links:
                    # print("contact_fail")
                    return False

            # The robot shouldn't be flipped, so limit the Roll and Pitch
            if self._current_base_ori_euler[0] > self.roll_threshold or self._current_base_ori_euler[0] < -self.roll_threshold:
                # print("roll_fail")
                return False
            if self._current_base_ori_euler[1] > self.pitch_threshold or self._current_base_ori_euler[1] < -self.pitch_threshold:
                # print("pitch_fail")
                return False

            # Issue - needs to account for heightfield data
            if self.enable_z_limit:
                # z_upper_limit = self._init_base_pos[2] + self.healthy_z_limit * 2
                # z_lower_limit = self._init_base_pos[2] - self.healthy_z_limit
                # # Random terrains don't have fixed heights in every run. Hence initial position of
                # # the robot is generally higher than usual. Taking that under consideration
                # if env.world_dict["ground"]["type"] != "plain":
                #     z_lower_limit -= 0.1
                # Robot shouldn't be below the limit
                if self._current_base_pos[2] < 0.02:
                    # print("z_fail: Current=", self._current_base_pos[2], ",\tLimit=", z_lower_limit)
                    return False
        return True

