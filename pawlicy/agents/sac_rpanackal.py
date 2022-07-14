from pawlicy.envs.a1_gym_env import A1GymEnv
from pawlicy.envs.gym_config import LocomotionGymConfig
from pawlicy.envs.wrappers import NormalizeActionWrapper

from pawlicy.robots import robot_config
from pawlicy.robots.a1 import constants as robot_const
from pawlicy.sensors import robot_sensors
from pawlicy.tasks.walk_along_x import WalkAlongX

from gym.wrappers import TimeLimit
import os
import inspect
from typing import Union
import argparse
import numpy as np
from absl import app
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

MOTOR_CONTROL_MODE_MAP = {
    'Torque': robot_config.MotorControlMode.TORQUE,
    'Position': robot_config.MotorControlMode.POSITION,
    'Hybrid': robot_config.MotorControlMode.HYBRID
}
SEED = 1

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
SAVE_DIR = os.path.join(parentdir, "saved")

class SACAgent():
    def __init__(self):
        
        self.gym_config = LocomotionGymConfig()
        self.env = self.setup_env()
        self.model = SAC('MlpPolicy', self.env, verbose=1, seed=SEED)
        pass


    def setup_env(self):
        
        self.gym_config.enable_rendering = True
        self.gym_config.motor_control_mode = MOTOR_CONTROL_MODE_MAP['Position']
        self.gym_config.reset_time = 2
        self.gym_config.num_action_repeat = 10
        self.gym_config.enable_action_interpolation = False
        self.gym_config.enable_action_filter = False
        self.gym_config.enable_clip_motor_commands = False
        self.gym_config.robot_on_rack = False
        self.gym_config.randomise_terrain = False
        self.gym_config.enable_hard_reset = True

        sensors = [
            robot_sensors.BaseDisplacementSensor(dtype=np.float32),
            robot_sensors.IMUSensor(dtype=np.float32),
            robot_sensors.MotorAngleSensor(num_motors=robot_const.NUM_MOTORS, dtype=np.float32),
        ]

        task = WalkAlongX()

        env = A1GymEnv(gym_config=self.gym_config, robot_sensors=sensors, task=task)
        #env = observation_dictionary_to_array_wrapper.ObservationDictionaryToArrayWrapper(env)
        # env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(env,
        #     trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(action_limit=-6.28318548203))

        # Normalize the action space
        env = NormalizeActionWrapper(env)
        # Set a time limit for each episode
        env = TimeLimit(env, max_episode_steps=100)
        # To monitor training stats
        env = Monitor(env)
        check_env(env, warn=True)
        # a simple vectorized wrapper
        env = DummyVecEnv([lambda: env])
        # Normalizes the observation space and rewards
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
        return env

    def learn(self):
        self.model.learn(total_timesteps=10000)
        print(f"Learning Complete.")
        self.save_model()

    def train(self):
        #self.learn()
        obs = self.env.reset()

        while True:
            #action, _state = self.model.predict(obs, deterministic=True)
            action = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(action)
            self.env.render()
            if done:
                obs = self.env.reset()

    def eval(self, n_episodes=20, name: Union[None, str] = None):
        if name is not None:
            self.model = self.load_model(name)

        for i in range(n_episodes):
            done = False
            while not done:
                action, _state = self.model.predict(obs, deterministic=True)
                #action = self.env.action_space.sample()
                obs, reward, done, info = self.env.step(action)
                self.env.render()
                if done:
                    obs = self.env.reset()


    def save_model(self, name="_sac_rpanackal"):
        self.model.save(os.path.join(SAVE_DIR, "models", name))
        self.model.save_replay_buffer(os.path.join(SAVE_DIR,"replay_buffers", name))
    
    def load_model(self, name="_sac_rpanackal"):
        self.model.load(os.path.join(SAVE_DIR, "models", name))
        self.model.load_replay_buffer(os.path.join(SAVE_DIR,"replay_buffers", name))


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
    agent.learn()
    agent.eval()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

