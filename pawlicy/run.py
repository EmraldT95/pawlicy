import os
import inspect
import argparse

from pawlicy.envs import A1PosEnv, A1TorEnv
from pawlicy.learning import Trainer
from pawlicy.tasks.walk_along_x_v8 import WalkAlongX

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
SAVE_DIR = os.path.join(currentdir, "agents")


def main():
    # Getting all the arguments passed
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--mode', "-m", dest="mode", default="test", choices=[
                            "train", "test"], type=str, help='to set to training or testing mode')
    arg_parser.add_argument("--motor_control_mode", "-mcm", dest="motor_control_mode", default="Torque",
                            choices=["Position", "Torque"], type=str, help="to set motor control mode")
    arg_parser.add_argument('--visualize', "-v", dest="visualize",
                            action="store_true", help='To flip rendering behaviour')
    arg_parser.add_argument("--randomise_terrain", "-rt", dest="randomise_terrain",
                            default=False, type=bool, help="to setup a randommized terrain")
    arg_parser.add_argument('--total_timesteps', "-tts", dest="total_timesteps",
                            default=int(1e6), type=int, help='total number of training steps')
    arg_parser.add_argument('--algorithm', "-a", dest="algorithm", default="SAC", choices=[
                            "SAC", "PPO", "TD3"], type=str, help='the algorithm used to train the robot')
    arg_parser.add_argument('--path', "-p", dest="path",
                            default='', type=str, help='the path to the saved model')
    arg_parser.add_argument('--max_episode_steps', "-mes", dest="max_episode_steps",
                            default=1000, type=int, help='maximum steps per episode')
    arg_parser.add_argument('--total_num_eps', "-tne", dest="total_num_eps",
                            default=20, type=int, help='total number of test episodes')
    arg_parser.add_argument('--load_final_model', "-lfm", dest="load_final_model",
                            action="store_true", help='Whether to load the final model instead of best model')
    args = arg_parser.parse_args()

    task = WalkAlongX()

    # Setting the save path
    if args.path != '':
        path = os.path.join(currentdir, args.path)
    else:
        path = os.path.join(SAVE_DIR, args.algorithm)

    # Training
    if args.mode == "train":
        if args.motor_control_mode == "Torque":
            env = A1TorEnv(args, args.visualize, task)
            eval_env = A1TorEnv(args, False, task)
        else:
            env = A1PosEnv(randomise_terrain=args.randomise_terrain,
                        motor_control_mode=args.motor_control_mode,
                        enable_rendering=args.visualize,
                        task=task)

            eval_env = A1PosEnv(randomise_terrain=args.randomise_terrain,
                                motor_control_mode=args.motor_control_mode,
                                enable_rendering=False,
                                task=task)

        # Get the trainer
        local_trainer = Trainer(env, args, eval_env, path)

        # The hyperparameters to override/add for the specific algorithm
        # (Check 'learning/hyperparams.yml' for default values)
        override_hyperparams = {
            "n_timesteps": args.total_timesteps,
            # "learning_rate_scheduler": "linear"
        }

        # Train the agent
        _ = local_trainer.train(override_hyperparams)

        # Save the model after training
        local_trainer.save_model()

    # Testing
    else:
        if args.motor_control_mode == "Torque":
            test_env = A1TorEnv(args, True, task)
        else:
            test_env = A1PosEnv(randomise_terrain=args.randomise_terrain,
                                motor_control_mode=args.motor_control_mode,
                                enable_rendering=True,
                                task=task)
        Trainer(test_env, args, save_path=path).test()


if __name__ == "__main__":
    main()
