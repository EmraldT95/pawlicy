import os
import inspect

from gym.wrappers import TimeLimit
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat

from pawlicy.envs.wrappers import NormalizeActionWrapper
from pawlicy.learning import utils

currentdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
SAVE_DIR = os.path.join(currentdir, "agents")

class Trainer:
    """
    The trainer class provides some basic methods to train an agent using different algorithms
    available in stable_baselines3

    Args:
        env: The gym environment to train on.
        eval_env: The environment to evaluate on
        algorithm: The algorithm to use.
        max_episode_steps: The no. of steps per episode
    """
    def __init__(self, env, eval_env=None, algorithm="SAC", max_episode_steps=100):
        self._eval_env = eval_env
        self._algorithm = algorithm
        self._max_episode_steps = max_episode_steps
        self._env = self.setup_env(env, self._max_episode_steps)
        # Setup the evaluation environment as well, if available
        if eval_env is not None:
            self._eval_env = self.setup_env(eval_env, self._max_episode_steps // 2)

    def train(self, override_hyperparams={}):
        """
        Trains an agent to use the environment to maximise the rewards while performing
        a specific task. This will tried out with multiple other algorithms later for
        benchmarking purposes.

        Args:
            override_hyperparams: The hyperparameters to override/add to the default config
        """
        # Get the default hyperparameters and override if needed
        _, hyperparameters = utils.read_hyperparameters(self._algorithm, 1, override_hyperparams)

        # Sanity checks
        n_timesteps = hyperparameters.pop("n_timesteps", None)
        if n_timesteps is None:
            raise ValueError("The hyperparameter 'n_timesteps' is missing.")
        eval_frequency = hyperparameters.pop("eval_freq", 5000)
        scheduler_type = hyperparameters.pop("learning_rate_scheduler", None)
        lr = hyperparameters.pop("learning_rate", float(1e-3))

        # Setup up learning rate scheduler arguments, if needed
        if scheduler_type is not None:
            lr_scheduler_args = {
                "lr_type": scheduler_type,
                "total_timesteps": n_timesteps
            }

        # Create the directory to store the logs in
        log_dir = os.path.join(SAVE_DIR, "logs")
        os.makedirs(log_dir, exist_ok=True)
        eval_log_dir = os.path.join(SAVE_DIR, "eval_logs")
        os.makedirs(eval_log_dir, exist_ok=True)

        # Check which algorithm to use
        if self._algorithm == "SAC":
            self._model = SAC(env=self._env,
                                verbose=1,
                                learning_rate=utils.lr_schedule(lr, **lr_scheduler_args) if scheduler_type is not None else lr,
                                tensorboard_log=log_dir,
                                **hyperparameters)

        # Train the model (check if evaluation is needed)
        if self._eval_env is not None:
            self._model.learn(n_timesteps,
                                log_interval=100,
                                eval_env=self._eval_env,
                                eval_freq=eval_frequency,
                                eval_log_path=eval_log_dir,
                                reset_num_timesteps=False,
                                callback=TensorboardCallback())
        else:
            self._model.learn(n_timesteps,
                                log_interval=100,
                                reset_num_timesteps=False,
                                callback=TensorboardCallback())

        # Return the trained model
        return self._model

    def save_model(self, save_path):
        """
        Saves the trained model. Also saves the replay buffer

        Args:
            model: The trained agent
        """
        if save_path is None:
            raise ValueError("No path specified to save the trained model.")
        else:
            # Create the directory to save the models in.
            os.makedirs(save_path, exist_ok=True)
            self._model.save(os.path.join(save_path, f"{self._algorithm}"))
            self._model.save_replay_buffer(os.path.join(save_path, f"{self._algorithm}_replay_buffer"))
            print(f"Model saved in path: {save_path}")

    def test(self, model_path=None):
        """
        Tests the agent

        Args:
            env: The gym environment to test the agent on.
        """
        if model_path is not None:
            self._model = SAC.load(os.path.join(model_path, f"{self._algorithm}"))
            self._model.load_replay_buffer(os.path.join(model_path, f"{self._algorithm}_replay_buffer"))

        obs = self._env.reset()
        for _ in range(500):
            action, _states = self._model.predict(obs, deterministic=True)
            obs, reward, done, info = self._env.step(action)
            if done:
                obs = self._env.reset()

    def setup_env(self, env, max_episode_steps):
        """
        Modifies the environment to suit to the needs of stable_baselines3.

        Args:
            max_episode_steps: The number of steps per episode
        """
        # Normalize the action space
        env = NormalizeActionWrapper(env)
        # Set the number of steps for each episode
        env = TimeLimit(env, max_episode_steps)
        # To monitor training stats
        env = Monitor(env)
        # a simple vectorized wrapper
        env = DummyVecEnv([lambda: env])
        # Normalizes the observation space and rewards
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

        return env;

    @property
    def model(self):
        return self._model

    @property
    def max_episode_steps(self):
        return self._max_episode_steps

    @max_episode_steps.setter
    def max_episode_steps(self, value):
        self._max_episode_steps = value

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        self._best_reward = 0
        super(TensorboardCallback, self).__init__(verbose)

    def _on_training_start(self):
        self._log_freq = 10000  # log every 10000 calls

        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        local_env = self.training_env.venv.envs[0]

        # Find the best reward
        reward = np.max(np.array(local_env.episode_returns)) if len(local_env.episode_returns) > 0 else 0
        self._best_reward = self._best_reward if self._best_reward > reward else reward

        if self.n_calls % self._log_freq == 0:
            self.tb_formatter.writer.add_scalar("x_position", local_env.robot.GetBasePosition()[0], self.num_timesteps)
            self.tb_formatter.writer.add_scalar("y_position", local_env.robot.GetBasePosition()[1], self.num_timesteps)
            self.tb_formatter.writer.add_scalar("z_position", local_env.robot.GetBasePosition()[2], self.num_timesteps)
            self.tb_formatter.writer.add_scalar("x_velocity", local_env.robot.GetBaseVelocity()[0], self.num_timesteps)
            self.tb_formatter.writer.add_scalar("best_reward", self._best_reward, self.num_timesteps)
            self.tb_formatter.writer.flush()