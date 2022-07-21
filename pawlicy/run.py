from pawlicy import learning
from pawlicy.envs import A1GymEnv, SampleEnv
from pawlicy.envs.gym_config import LocomotionGymConfig
from pawlicy.robots import robot_config
from pawlicy.robots.a1 import constants
from pawlicy.sensors import robot_sensors
from pawlicy.tasks import walk_along_x
from pawlicy.learning import Trainer, utils
from pawlicy.utils import env_utils

import os
import inspect
import argparse
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
SAVE_DIR = os.path.join(currentdir, "agents")

MOTOR_CONTROL_MODE_MAP = {
    'Torque': robot_config.MotorControlMode.TORQUE,
    'Position': robot_config.MotorControlMode.POSITION,
    'Hybrid': robot_config.MotorControlMode.HYBRID
}

def build_env(randomise_terrain, motor_control_mode, enable_rendering):
    """ Builds the gym environment needed for RL

    Args:
        randomise_terrain: Whether to randomize terrain or not
        motor_control_mode: Position, Torque or Hybrid
        enable_rendering: Whether to configure pybullet in GUI mode or DIRECT mode
    """
    gym_config = LocomotionGymConfig()
    gym_config.enable_rendering = enable_rendering
    gym_config.motor_control_mode = motor_control_mode
    gym_config.reset_time = 2
    gym_config.num_action_repeat = 5
    gym_config.enable_action_interpolation = True
    gym_config.enable_action_filter = False
    gym_config.enable_clip_motor_commands = True
    gym_config.randomise_terrain = randomise_terrain

    task = walk_along_x.WalkAlongX()

    sensors = [
        robot_sensors.BaseDisplacementSensor(dtype=np.float32),
        robot_sensors.IMUSensor(dtype=np.float32),
        robot_sensors.MotorAngleSensor(num_motors=constants.NUM_MOTORS, dtype=np.float32),
    ]

    env = A1GymEnv(gym_config=gym_config, robot_sensors=sensors, task=task)
    # env = SampleEnv(enable_rendering,
    #                 randomise_terrain=randomise_terrain,
    #                 motor_control_mode=motor_control_mode,
    #                 task=task)

    return env

def main():

    arg_parser = argparse.ArgumentParser()
    # arg_parser.add_argument('--log_lvl', "-ll", dest="log_lvl", default="DEBUG", type=str, help='set log level')
    arg_parser.add_argument('--visualize', "-v", dest="visualize", action="store_true", help='To flip rendering behaviour')
    arg_parser.add_argument("--motor_control_mode", "-mcm", dest="motor_control_mode",  default="Position", choices=["Position", "Torque", "Velocity"],
        type=str, help="to set motor control mode")
    arg_parser.add_argument("--randomise_terrain", "-rt", dest="randomise_terrain", default=False, type=bool, help="to setup a randommized terrain")
    arg_parser.add_argument('--total_timesteps', "-tts", dest="total_timesteps", default=int(1e6), type=int, help='total number of training steps')
    arg_parser.add_argument('--mode', "-m", dest="mode", default="test", choices=["train", "test"], type=str, help='to set to training or testing mode')
    args = arg_parser.parse_args()

    # Set the logger level
    # env_utils._set_log_lvl(args.log_lvl)

    # Training
    if args.mode == "train":
        env = build_env(randomise_terrain=args.randomise_terrain,
                    motor_control_mode=MOTOR_CONTROL_MODE_MAP[args.motor_control_mode],
                    enable_rendering=args.visualize)

        # Need to do this because our current pybullet setup can have only one client with GUI enabled
        if args.visualize:
            eval_env = None
        else:
            eval_env = build_env(randomise_terrain=args.randomise_terrain,
                        motor_control_mode=MOTOR_CONTROL_MODE_MAP[args.motor_control_mode],
                        enable_rendering=args.visualize)

        # Get the trainer
        local_trainer = Trainer(env, eval_env, "SAC")

        # The hyperparameters to override/add for the specific algorithm
        # (Check 'learning/hyperparams.yml' for default values)
        override_hyperparams = {
            "n_timesteps": args.total_timesteps,
            "learning_starts": 1000,
        }

        # Train the agent
        _ = local_trainer.train(override_hyperparams)

        # Save the model after training
        local_trainer.save_model(SAVE_DIR)

    # Testing
    else:
        test_env = build_env(randomise_terrain=args.randomise_terrain,
                    motor_control_mode=MOTOR_CONTROL_MODE_MAP[args.motor_control_mode],
                    enable_rendering=True)        
        Trainer(test_env, algorithm="SAC").test(SAVE_DIR)

if __name__ == "__main__":
    main()
