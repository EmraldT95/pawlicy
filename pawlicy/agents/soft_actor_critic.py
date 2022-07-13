
from pawlicy.envs.a1_gym_env import A1GymEnv
from pawlicy.envs.gym_config import LocomotionGymConfig
from pawlicy.envs.wrappers import observation_dictionary_to_array_wrapper
from pawlicy.robots import robot_config
from pawlicy.robots.a1 import constants as robot_const
from pawlicy.sensors import robot_sensors
from pawlicy.tasks.walk_along_x import WalkAlongX

import numpy as np
from absl import app
from absl import flags
from stable_baselines3 import SAC

FLAGS = flags.FLAGS
flags.DEFINE_enum('robot_type', 'A1', ['A1', 'Laikago'], 'Robot Type.')
flags.DEFINE_enum('motor_control_mode', 'Torque',
                  ['Torque', 'Position', 'Hybrid'], 'Motor Control Mode.')
flags.DEFINE_bool('on_rack', False, 'Whether to put the robot on rack.')
flags.DEFINE_string('video_dir', None,

                    'Where to save video (or None for not saving).')
MOTOR_CONTROL_MODE_MAP = {
    'Torque': robot_config.MotorControlMode.TORQUE,
    'Position': robot_config.MotorControlMode.POSITION,
    'Hybrid': robot_config.MotorControlMode.HYBRID
}

class SACAgent():
    def __init__(self):
        
        self.env = self.setup_env()
        self.model = SAC('MlpPolicy', self.env, verbose=1)
        pass


    def setup_env(self):
        gym_config = LocomotionGymConfig()
        gym_config.enable_rendering = True
        gym_config.motor_control_mode = MOTOR_CONTROL_MODE_MAP[FLAGS.motor_control_mode]
        gym_config.reset_time = 2
        gym_config.num_action_repeat = 10
        gym_config.enable_action_interpolation = False
        gym_config.enable_action_filter = False
        gym_config.enable_clip_motor_commands = False
        gym_config.robot_on_rack = False
        gym_config.randomise_terrain = False

        sensors = [
            robot_sensors.BaseDisplacementSensor(),
            robot_sensors.IMUSensor(),
            robot_sensors.MotorAngleSensor(num_motors=robot_const.NUM_MOTORS),
        ]

        task = WalkAlongX(healthy_roll_limit=np.pi / 2)

        env = A1GymEnv(gym_config=gym_config, robot_sensors=sensors, task=task)
        env = observation_dictionary_to_array_wrapper.ObservationDictionaryToArrayWrapper(env)

        return env

    def train(self):
        #self.model.learn(total_timesteps=10000)
        
        #print(f"Learning Complete.")
        #obs = self.env.reset()
        for i in range(1000):
            #action, _state = self.model.predict(obs, deterministic=True)
            action = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(action)
            self.env.render()
            if done:
                obs = self.env.reset()

def main(_):
    agent = SACAgent()
    agent.train()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass