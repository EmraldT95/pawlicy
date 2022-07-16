from pawlicy.envs import A1GymEnv
from pawlicy.envs.gym_config import LocomotionGymConfig
from pawlicy.robots import robot_config
from pawlicy.robots.a1 import constants
from pawlicy.sensors import robot_sensors
from pawlicy.tasks import walk_along_x
from pawlicy.learning import Trainer, utils

import os
import inspect
import argparse
import numpy as np
from typing import Union

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
SAVE_DIR = os.path.join(currentdir, "agents")

MOTOR_CONTROL_MODE_MAP = {
    'Torque': robot_config.MotorControlMode.TORQUE,
    'Position': robot_config.MotorControlMode.POSITION,
    'Hybrid': robot_config.MotorControlMode.HYBRID
}

def build_env(args, enable_rendering=False):
    """ Builds the gym environment needed for RL

    Args:
        randomise_terrain: Whether to randomize terrain or not
        motor_control_mode: Position, Torque or Hybrid
        enable_rendering: Whether to configure pybullet in GUI mode or DIRECT mode
        robot_on_rack: Whether robot is on rack or not
    """
    gym_config = LocomotionGymConfig()
    gym_config.enable_rendering = enable_rendering
    gym_config.motor_control_mode = MOTOR_CONTROL_MODE_MAP[args.motor_control_mode]
    gym_config.reset_time = 2
    gym_config.num_action_repeat = 10
    gym_config.enable_action_interpolation = True
    gym_config.enable_action_filter = True
    gym_config.enable_clip_motor_commands = False
    gym_config.robot_on_rack = False
    gym_config.randomise_terrain = args.randomise_terrain

    if args.visualize:
        gym_config.enable_rendering = args.visualize

    task = walk_along_x.WalkAlongX()

    sensors = [
        robot_sensors.BaseDisplacementSensor(dtype=np.float32),
        robot_sensors.IMUSensor(dtype=np.float32),
        robot_sensors.MotorAngleSensor(num_motors=constants.NUM_MOTORS, dtype=np.float32),
    ]

    env = A1GymEnv(gym_config=gym_config, robot_sensors=sensors, task=task)

    return env

def parse_arguements():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', "-m", dest="mode", default="test", choices=["train", "test"], type=str, help='to set to training or testing mode')
    parser.add_argument('--max_episode_steps', "-mes", dest="max_episode_steps", default=1000, type=int, help='maximum steps per episode')
    parser.add_argument('--visualize', "-v", dest="visualize", default=None, type=Union[bool, None], help='To override rendering behaviour')
    parser.add_argument("--randomise_terrain", "-rt", dest="randomise_terrain", default=False, type=bool, help="to setup a randommized terrain")
    parser.add_argument("--motor_control_mode", "-mcm", dest="motor_control_mode",  default="position", choices=["position", "torque", "hybrid"], type=str, help="to set motor control mode")

    parser.add_argument('--author', "-au", dest="author", default="rpanackal", type=str, help='name of author')
    parser.add_argument('--exp_suffix', "-s", dest="exp_suffix", default="", type=str, help='appends to experiment name')
    parser.add_argument('--total_timesteps', "-tts", dest="total_timesteps", default=int(1e5), type=int, help='total number of training steps')
    
    parser.add_argument('--total_num_eps', "-tne", dest="total_num_eps", default=20, type=int, help='total number of test episodes')
    parser.add_argument('--load_exp_name', "-l", dest="load_exp_name", default="sac_rpanackal_tns100000", type=str, help='name of experiment to be tested')
    #parser.add_argument('--mode', "-m", default="eval", choices=["train", "eval"], type=str, help='To set to training or evaluation mode')

    args = parser.parse_args()
    
    args.motor_control_mode = args.motor_control_mode.capitalize()
    return args

def main():

    args = parse_arguements()

    # Training
    if args.mode == "train":
        env = build_env(args, enable_rendering=False)

        # Train the agent
        local_trainer = Trainer(env, "SAC", args)
        _, hyperparameters = utils.read_hyperparameters("SAC", 1, {"learning_starts": 2000})
        model = local_trainer.train(hyperparameters)

        # Save the model after training
        local_trainer.save_model(SAVE_DIR)

    # Testing
    if args.mode == "test":
        test_env = build_env(args, enable_rendering=True)
        Trainer(test_env, "SAC", args).test(SAVE_DIR)

if __name__ == "__main__":
    main()
