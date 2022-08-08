import os
import subprocess
import glob
import inspect
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import pybullet as p
import numpy as np
import yaml
import matplotlib.pyplot as plt
import gym
from typing import Callable, Union, Tuple, Dict, Any
from stable_baselines3.common import base_class
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped, sync_envs_normalization

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

def lr_schedule(initial_value: Union[float, str],
                lr_type: str,
                total_timesteps: Union[int, None] = None,
                final_value: Union[float, str] = 1e5,) -> Callable[[float], float]:
    """
    Learning rate scheduler that is configured.
    
    Args:
        initial_value: The initial learning rate
        lr_type: The scheduler type
        total_timesteps: The total timesteps to train the agent
    Returns: 
        (function): the scheduler function
    """
    lr_type = lr_type
    iters = np.arange(total_timesteps)
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
    
    if lr_type == "cosine":
        if isinstance(final_value, str):
            final_value = float(final_value)
        schedule = final_value + 0.5 * (initial_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    def func(progress_remaining: float) -> float:
        """
        The new learning rate
        Args:
            progress_remaining: The progress remaining - will decrease from 1 (beginning) to 0
        Returns:
            (float)
        """
        # Cosine Annealing
        if lr_type == "cosine":
            idx = int((1 - progress_remaining) *  (total_timesteps - 1))
            return schedule[idx]
        # Linear
        else:
            return np.max(progress_remaining * initial_value, int(1e-5))

    return func

def read_hyperparameters(algorithm: str, verbose=0, custom_hyperparams=None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Reads the default hyperparameter config for the given algorithm from
    a common YAML file. These can be overriden using the custom_hyperparams argument.

    Args:
        algorithm: The algorithm for which to get the hyperparameters
        verbose: Whether to print the final hyperparameters in the console or not
        custom_hyperparams: The hyperparameters to change/add
    """
    # Load hyperparameters from yaml file
    file_path = os.path.join(currentdir, "hyperparams.yml")
    with open(file_path) as f:
        hyperparams_dict = yaml.safe_load(f)
        # Find the correct hyperparameters based on the keys
        if algorithm in list(hyperparams_dict.keys()):
            hyperparams = hyperparams_dict[algorithm]
        else:
            raise ValueError(f"Hyperparameters not found for {algorithm}")

    if custom_hyperparams is not None:
        # Overwrite hyperparams if needed
        hyperparams.update(custom_hyperparams)
    # Sort hyperparams that will be saved
    saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])

    if verbose > 0:
        print("Default hyperparameters for environment (ones being tuned will be overridden):")
        print(saved_hyperparams)

    return hyperparams, saved_hyperparams

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0, mode="Torque"):
        self._best_reward = 0
        self._mode = mode

        if self._mode == "Torque":
            self.max_hip_torque = float("-inf")
            self.min_hip_torque = float("inf")
            self.max_leg_torque = float("-inf")
            self.min_leg_torque = float("inf")
            self.max_hip_velocity = float("-inf")
            self.min_hip_velocity = float("inf")
            self.max_leg_velocity = float("-inf")
            self.min_leg_velocity = float("inf")

        super(TensorboardCallback, self).__init__(verbose)

    def _on_training_start(self):
        self._log_freq = 10000  # log every 10000 calls

        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        local_env = self.training_env.venv.envs[0]

        if self._mode == "Torque":
            self._set_max_min_veocities(local_env)
            self._set_max_min_torques(local_env)

        # Find the best reward
        reward = np.max(np.array(local_env.episode_returns)) if len(local_env.episode_returns) > 0 else 0
        self._best_reward = self._best_reward if self._best_reward > reward else reward

        if self.n_calls % self._log_freq == 0:
            self.tb_formatter.writer.add_scalar("x_position", local_env.robot.GetBasePosition()[0], self.num_timesteps)
            self.tb_formatter.writer.add_scalar("y_position", local_env.robot.GetBasePosition()[1], self.num_timesteps)
            self.tb_formatter.writer.add_scalar("z_position", local_env.robot.GetBasePosition()[2], self.num_timesteps)
            self.tb_formatter.writer.add_scalar("x_velocity", local_env.robot.GetBaseVelocity()[0], self.num_timesteps)
            self.tb_formatter.writer.add_scalar("best_reward", self._best_reward, self.num_timesteps)
            if self._mode == "Torque":
                self.tb_formatter.writer.add_scalar("max_hip_torque", self.max_hip_torque, self.num_timesteps)
                self.tb_formatter.writer.add_scalar("min_hip_torque", self.min_hip_torque, self.num_timesteps)
                self.tb_formatter.writer.add_scalar("max_leg_torque", self.max_leg_torque, self.num_timesteps)
                self.tb_formatter.writer.add_scalar("min_leg_torque", self.min_leg_torque, self.num_timesteps)

                self.tb_formatter.writer.add_scalar("max_hip_velocity", self.max_hip_velocity, self.num_timesteps)
                self.tb_formatter.writer.add_scalar("min_hip_velocity", self.min_hip_velocity, self.num_timesteps)
                self.tb_formatter.writer.add_scalar("max_leg_velocity", self.max_leg_velocity, self.num_timesteps)
                self.tb_formatter.writer.add_scalar("min_leg_velocity", self.min_leg_velocity, self.num_timesteps)
            self.tb_formatter.writer.flush()

    def _set_max_min_torques(self, local_env):
        joint_torques = local_env.robot.GetMotorTorques()
        for i, torque in enumerate(joint_torques):
            if i % 3 == 0:
                if torque > self.max_hip_torque:
                    self.max_hip_torque = torque
                if torque < self.min_hip_torque:
                    self.min_hip_torque = torque
            else:
                if torque > self.max_leg_torque:
                    self.max_leg_torque = torque
                if torque < self.min_leg_torque:
                    self.min_leg_torque = torque
    
    def _set_max_min_veocities(self, local_env):
        joint_veocities = local_env.robot.GetMotorVelocities()
        for i, velocity in enumerate(joint_veocities):
            if i % 3 == 0:
                if velocity > self.max_hip_velocity:
                    self.max_hip_velocity = velocity
                if velocity < self.min_hip_velocity:
                    self.min_hip_velocity = velocity
            else:
                if velocity > self.max_leg_velocity:
                    self.max_leg_velocity = velocity
                if velocity < self.min_leg_velocity:
                    self.min_leg_velocity = velocity
    
    def _set_max_min_foot_positions(self, local_env):
        foot_pos = local_env.robot.GetFootPositionsInBaseFrame().ravel()
        for i, pos in enumerate(foot_pos):
                if pos > self.max_foot_pos:
                    self.max_foot_pos = pos
                if pos < self.min_foot_pos:
                    self.min_foot_pos = pos

class Eval_Callback(EvalCallback):
    """Callback function used in the evaluation process"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._var_threshold = 1/5 # The max allowed deviation from the mean
        # The video saving part
        if "callback_after_eval" in kwargs.keys():
            self._save_video = True
            os.makedirs(os.path.join(self.best_model_save_path, "videos"), exist_ok=True)
        else:
            self._save_video = False

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)
        # Saving the image plots if the evaluation is called at every 10000th step of training
        if self._save_video and self.num_timesteps % 10000 == 0:
            # Save every 5th image
            if self._save_plot_iteration % 5 == 0:
                plt.imshow(self.eval_env.render())
                plt.savefig(os.path.join(self.best_model_save_path, f'videos/plot_{self._plot_iteration}.png'))
                self._plot_iteration += 1
            self._save_plot_iteration += 1

class After_Eval_Callback(BaseCallback):
    """Call back function called after the evaluation is complete"""
    
    def __init__(self, verbose: int = 0, best_model_save_path = None):
        self.best_model_save_path = best_model_save_path
        super().__init__(verbose)

    def _on_step(self) -> bool:
        super()._on_step()
        if int(self.num_timesteps % 10000) == 0:
            self.generate_video()

    def generate_video(self):
        """Creates a video using the plots"""
        os.chdir(os.path.join(self.best_model_save_path, "videos"))
        subprocess.call([
            'ffmpeg', '-i', 'plot_%d.png', '-r', '30', f'model_{self.num_timesteps}.mp4'
        ])
        for file_name in glob.glob("*.png"):
            os.remove(file_name)
