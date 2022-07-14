import os
import inspect
import numpy as np

from gym.wrappers import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch

from pawlicy.envs.a1_gym_env import A1GymEnv
from pawlicy.robots import robot_config
from pawlicy.robots.a1 import constants
from pawlicy.envs.gym_config import LocomotionGymConfig
from pawlicy.envs.wrappers import NormalizeActionWrapper
#         time_limit_wrapper, \
#         observation_dictionary_to_array_wrapper, \
#         trajectory_generator_wrapper_env, \
#         simple_openloop
from pawlicy.sensors import robot_sensors
from pawlicy.tasks import walk_along_x

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

SAVE_DIR = os.path.join(parentdir, "saved/models/sac")

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
    gym_config.num_action_repeat = 10
    gym_config.enable_action_interpolation = False
    gym_config.enable_action_filter = False
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
    # env = observation_dictionary_to_array_wrapper.ObservationDictionaryToArrayWrapper(env)
    # This will work only for position
    # env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(env,
    #     trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(action_limit=-6.28318548203))

    return env

def train(env, eval_env=None, lr=1e-3, bs=64):
    # Load the model.
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=lr,
        buffer_size=int(1e6),
        gamma=0.95,
        batch_size=bs,
        device="cuda")

    # Train the model
    if eval_env is not None:
        model.learn(total_timesteps=500, log_interval=4, eval_env=eval_env, eval_freq=100)
    else:
        model.learn(total_timesteps=500, log_interval=4)
    return model

def test():
    raise NotImplementedError

if __name__ == "__main__":
    # Create the directory to save the models in.
    os.makedirs(SAVE_DIR, exist_ok=True)

    env = build_env(randomise_terrain=False,
                    motor_control_mode=MOTOR_CONTROL_MODE_MAP['Position'],
                    enable_rendering=False)

    # eval_env = build_env(randomise_terrain=False,
    #                 motor_control_mode=MOTOR_CONTROL_MODE_MAP['Position'],
    #                 enable_rendering=False)

    # # sample an observation from the environment
    # obs = model.env.observation_space.sample()

    # Normalize the action space
    env = NormalizeActionWrapper(env)
    # Set a time limit for each episode
    env = TimeLimit(env, max_episode_steps=100)
    # To monitor training stats
    env = Monitor(env)
    # a simple vectorized wrapper
    env = DummyVecEnv([lambda: env])
    # Normalizes the observation space and rewards
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # eval_env = NormalizeActionWrapper(eval_env)
    # eval_env = Monitor(eval_env)
    # eval_env = DummyVecEnv([lambda: eval_env])
    # eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    model = train(env)
    model.save(os.path.join(SAVE_DIR, "sac_emrald"))
    model.save_replay_buffer(os.path.join(SAVE_DIR, "replay_buffer_emrald"))
    # # Check prediction before saving
    # print("pre saved", model.predict(obs, deterministic=True))