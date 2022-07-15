from faulthandler import enable
from pawlicy.envs.a1_gym_env import A1GymEnv
from pawlicy.envs.gym_config import LocomotionGymConfig
from pawlicy.envs.wrappers import NormalizeActionWrapper

from pawlicy.robots import robot_config
from pawlicy.robots.a1 import constants as robot_const
from pawlicy.sensors import robot_sensors
from pawlicy.tasks.walk_along_x import WalkAlongX

from gym.wrappers import TimeLimit
import os
import pathlib
from typing import Union
import argparse
import numpy as np
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

path = pathlib.Path(__file__)
SAVE_DIR = path.parents[1].resolve().joinpath("saved/models").as_posix()
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# os.sys.path.insert(0, parentdir)
# SAVE_DIR = os.path.join(parentdir, "saved")

def parse_arguements():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--mode', "-m", default="valid", choices=["train", "valid"], type=str, help='To set to training or validation mode')
    parser.add_argument('--steps_per_ep', "-spe", default=10000, type=int, help='maximum steps per episode')
    parser.add_argument('--render_mode', "-rm", default=None, type=Union[bool, None], help='To override rendering behaviour')

    parser.add_argument('--author', "-au", default="rpanackal", type=str, help='name of author')
    parser.add_argument('--exp_suffix', "-s", default="", type=str, help='appends to experiment name')
    parser.add_argument('--total_num_steps', "-tns", default=100000, type=int, help='total number of training steps')
    
    parser.add_argument('--total_num_eps', "-tne", default=20, type=int, help='total number of validation episodes')
    parser.add_argument('--load_exp_name', "-l", default="sac_rpanackal_tns100000", type=str, help='name of experiment to be validated')
    #parser.add_argument('--mode', "-m", default="eval", choices=["train", "eval"], type=str, help='To set to training or evaluation mode')

    args = parser.parse_args()
    #args.log = args.log.upper()
    return args

def setup_env(args, enable_rendering=False):

    gym_config = LocomotionGymConfig()
    gym_config.enable_rendering = enable_rendering
    gym_config.motor_control_mode = MOTOR_CONTROL_MODE_MAP['Position']
    gym_config.reset_time = 2
    gym_config.num_action_repeat = 10
    gym_config.enable_action_interpolation = False
    gym_config.enable_action_filter = False
    gym_config.enable_clip_motor_commands = False
    gym_config.robot_on_rack = False
    gym_config.randomise_terrain = False
    gym_config.enable_hard_reset = True

    if args.render_mode:
        gym_config.enable_rendering = args.render_mode

    sensors = [
        robot_sensors.BaseDisplacementSensor(dtype=np.float32),
        robot_sensors.IMUSensor(dtype=np.float32),
        robot_sensors.MotorAngleSensor(num_motors=robot_const.NUM_MOTORS, dtype=np.float32),
    ]

    task = WalkAlongX()

    env = A1GymEnv(gym_config=gym_config, robot_sensors=sensors, task=task)
    #env = observation_dictionary_to_array_wrapper.ObservationDictionaryToArrayWrapper(env)
    # env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(env,
    #     trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(action_limit=-6.28318548203))

    # Normalize the action space
    env = NormalizeActionWrapper(env)
    # Set a time limit for each episode
    env = TimeLimit(env, max_episode_steps=args.steps_per_ep)
    # To monitor training stats
    env = Monitor(env)
    check_env(env, warn=True)
    # a simple vectorized wrapper
    env = DummyVecEnv([lambda: env])
    # Normalizes the observation space and rewards
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    return env

def train(args,
        enable_rendering: bool = False):

    total_timesteps = args.total_num_steps

    env = setup_env(args, enable_rendering=False)
    model = SAC('MlpPolicy', env, verbose=1, seed=SEED)

    print(f"Learning in progress...")
    model.learn(total_timesteps=total_timesteps)
    print(f"Learning Complete.")

    exp_name = f"sac_{args.author}_tns{total_timesteps}"
    if args.exp_suffix:
        exp_name = f"{exp_name}_{args.exp_suffix}"

    model.save(os.path.join(SAVE_DIR, exp_name, "model"))
    model.save_replay_buffer(os.path.join(SAVE_DIR, exp_name, "replay_buffer"))

def valid(args,
        enable_rendering: bool = True):
    
    env = setup_env(args, enable_rendering=True)
    model = SAC.load(os.path.join(SAVE_DIR, args.load_exp_name, "model"))

    for i in range(args.total_num_eps):
        done = False
        obs = env.reset()
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            #action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            env.render()

    # model.load(os.path.join(SAVE_DIR, args.load_exp_name, "model"))
    # model.load_replay_buffer(os.path.join(SAVE_DIR, args.load_exp_name, "replay_buffer"))

if __name__ == '__main__':
    args = parse_arguements()
    
    try:
        if args.mode == "train":
            train(args)
        else:
            valid(args)

    except SystemExit:
        pass

