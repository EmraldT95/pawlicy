
from pawlicy.envs.a1_gym_env import A1GymEnv
from pawlicy.envs.gym_config import LocomotionGymConfig
from pawlicy.envs.wrappers import observation_dictionary_to_array_wrapper, trajectory_generator_wrapper_env, simple_openloop
from pawlicy.robots import robot_config
from pawlicy.robots.a1 import constants as robot_const
from pawlicy.sensors import robot_sensors
from pawlicy.tasks.walk_along_x import WalkAlongX

import argparse
import numpy as np
from absl import app
from absl import flags
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

FLAGS = flags.FLAGS
flags.DEFINE_enum('robot_type', 'A1', ['A1', 'Laikago'], 'Robot Type.')
flags.DEFINE_enum('motor_control_mode', 'Position',
                  ['Torque', 'Position', 'Hybrid'], 'Motor Control Mode.')
flags.DEFINE_bool('on_rack', False, 'Whether to put the robot on rack.')
flags.DEFINE_string('video_dir', None,

                    'Where to save video (or None for not saving).')
MOTOR_CONTROL_MODE_MAP = {
    'Torque': robot_config.MotorControlMode.TORQUE,
    'Position': robot_config.MotorControlMode.POSITION,
    'Hybrid': robot_config.MotorControlMode.HYBRID
}
SEED = 1

class SACAgent():
    def __init__(self):
        
        self.gym_config = LocomotionGymConfig()
        self.env = self.setup_env()
        self.model = SAC('MlpPolicy', self.env, verbose=1, seed=SEED)
        pass


    def setup_env(self):
        
        self.gym_config.enable_rendering = True
        self.gym_config.motor_control_mode = MOTOR_CONTROL_MODE_MAP[FLAGS.motor_control_mode]
        self.gym_config.reset_time = 2
        self.gym_config.num_action_repeat = 10
        self.gym_config.enable_action_interpolation = False
        self.gym_config.enable_action_filter = False
        self.gym_config.enable_clip_motor_commands = False
        self.gym_config.robot_on_rack = False
        self.gym_config.randomise_terrain = False
        self.gym_config.enable_hard_reset = True

        sensors = [
            robot_sensors.BaseDisplacementSensor(),
            robot_sensors.IMUSensor(),
            robot_sensors.MotorAngleSensor(num_motors=robot_const.NUM_MOTORS),
        ]

        task = WalkAlongX(healthy_roll_limit=np.pi / 2)

        env = A1GymEnv(gym_config=self.gym_config, robot_sensors=sensors, task=task)
        #env = observation_dictionary_to_array_wrapper.ObservationDictionaryToArrayWrapper(env)
        # env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(env,
        #     trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(action_limit=-6.28318548203))

        check_env(env, warn=True)
        
        return env

    def learn(self):
        self.model.learn(total_timesteps=10000)


    def train(self):
        #self.learn()
        
        print(f"Learning Complete.")
        obs = self.env.reset()

        while True:
            #action, _state = self.model.predict(obs, deterministic=True)
            action = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(action)
            self.env.render()
            if done:
                obs = self.env.reset()


def parse_arguements():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--log', "-l", default="debug", type=str, help='set log level')
    parser.add_argument('--log', "-l", default="debug", type=str, help='set log level')
    parser.add_argument('--log', "-l", default="debug", type=str, help='set log level')
    parser.add_argument('--log', "-l", default="debug", type=str, help='set log level')
    parser.add_argument('--log', "-l", default="debug", type=str, help='set log level')
    parser.add_argument('--log', "-l", default="debug", type=str, help='set log level')
    parser.add_argument('--log', "-l", default="debug", type=str, help='set log level')
    parser.add_argument('--log', "-l", default="debug", type=str, help='set log level')
    parser.add_argument('--log', "-l", default="debug", type=str, help='set log level')
    parser.add_argument('--log', "-l", default="debug", type=str, help='set log level')
    parser.add_argument('--log', "-l", default="debug", type=str, help='set log level')
    parser.add_argument('--log', "-l", default="debug", type=str, help='set log level')
    parser.add_argument('--log', "-l", default="debug", type=str, help='set log level')
    parser.add_argument('--log', "-l", default="debug", type=str, help='set log level')
    parser.add_argument('--log', "-l", default="debug", type=str, help='set log level')
    parser.add_argument('--log', "-l", default="debug", type=str, help='set log level')
    parser.add_argument('--log', "-l", default="debug", type=str, help='set log level')
    # parser.add_argument('--sum', dest='accumulate', action='store_const',
    #                     const=sum, default=max,
    #                     help='sum the integers (default: find the max)')


    args = parser.parse_args()
    #args.log = args.log.upper()
    return args

def main(_):
    agent = SACAgent()
    agent.train()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

