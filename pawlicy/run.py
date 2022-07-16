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
        robot_on_rack: Whether robot is on rack or not
    """
    gym_config = LocomotionGymConfig()
    gym_config.enable_rendering = enable_rendering
    gym_config.motor_control_mode = motor_control_mode
    gym_config.reset_time = 2
    gym_config.num_action_repeat = 10
    gym_config.enable_action_interpolation = True
    gym_config.enable_action_filter = True
    gym_config.enable_clip_motor_commands = False
    gym_config.robot_on_rack = False
    gym_config.randomise_terrain = randomise_terrain

    task = walk_along_x.WalkAlongX()

    sensors = [
        robot_sensors.BaseDisplacementSensor(dtype=np.float32),
        robot_sensors.IMUSensor(dtype=np.float32),
        robot_sensors.MotorAngleSensor(num_motors=constants.NUM_MOTORS, dtype=np.float32),
    ]

    env = A1GymEnv(gym_config=gym_config, robot_sensors=sensors, task=task)

    return env

def main():

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--visualize", dest="visualize", type=bool, default=False)
    arg_parser.add_argument("--mode", dest="mode", type=str, default="train")
    arg_parser.add_argument("--randomise_terrain", dest="randomise_terrain", type=bool, default=False)
    arg_parser.add_argument("--total_timesteps", dest="total_timesteps", type=int, default=20000)

    args = arg_parser.parse_args()

    # Training
    if args.mode == "train":
        env = build_env(randomise_terrain=args.randomise_terrain,
                    motor_control_mode=MOTOR_CONTROL_MODE_MAP["Position"],
                    enable_rendering=args.visualize)

        # Train the agent
        local_trainer = Trainer(env, "SAC")
        _, hyperparameters = utils.read_hyperparameters("SAC", 1, {"learning_starts": 2000})
        model = local_trainer.train(hyperparameters, args.total_timesteps)

        # Save the model after training
        local_trainer.save_model(SAVE_DIR)

    # Testing
    if args.mode == "test":
        test_env = build_env(randomise_terrain=args.randomise_terrain,
                    motor_control_mode=MOTOR_CONTROL_MODE_MAP["Position"],
                    enable_rendering=args.visualize)
        Trainer(test_env, "SAC").test(SAVE_DIR)

if __name__ == "__main__":
    main()
