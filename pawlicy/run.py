"""Simple script for executing random actions on A1 robot."""


from pawlicy.envs import a1_gym_env
from pawlicy.envs.gym_config import SimulationParameters, LocomotionGymConfig
from pawlicy.robots import robot_config
from pawlicy.robots.a1 import constants
from pawlicy.envs.wrappers import observation_dictionary_to_array_wrapper, trajectory_generator_wrapper_env, simple_openloop
from pawlicy.sensors import robot_sensors

from absl import app
from absl import flags
import numpy as np
from tqdm import tqdm
import pybullet as p  # pytype: disable=import-error


FLAGS = flags.FLAGS
flags.DEFINE_enum('robot_type', 'A1', ['A1', 'Laikago'], 'Robot Type.')
flags.DEFINE_enum('motor_control_mode', 'Torque',
                  ['Torque', 'Position', 'Hybrid'], 'Motor Control Mode.')
flags.DEFINE_bool('on_rack', False, 'Whether to put the robot on rack.')
flags.DEFINE_string('video_dir', None,
                    'Where to save video (or None for not saving).')

# ROBOT_CLASS_MAP = {'A1': a1.A1, 'Laikago': laikago.Laikago}

MOTOR_CONTROL_MODE_MAP = {
    'Torque': robot_config.MotorControlMode.TORQUE,
    'Position': robot_config.MotorControlMode.POSITION,
    'Hybrid': robot_config.MotorControlMode.HYBRID
}

def main(_):
    sim_params = SimulationParameters()
    sim_params.enable_rendering = True
    sim_params.motor_control_mode = MOTOR_CONTROL_MODE_MAP[FLAGS.motor_control_mode]
    sim_params.reset_time = 2
    sim_params.num_action_repeat = 10
    sim_params.enable_action_interpolation = False
    sim_params.enable_action_filter = False
    sim_params.enable_clip_motor_commands = False
    sim_params.robot_on_rack = False

    gym_config = LocomotionGymConfig(simulation_parameters=sim_params)
    sensors = [
        robot_sensors.BaseDisplacementSensor(),
        robot_sensors.IMUSensor(),
        robot_sensors.MotorAngleSensor(num_motors=constants.NUM_MOTORS),
    ]

    env = a1_gym_env.A1GymEnv(gym_config=gym_config, robot_sensors=sensors, randomize_terrain=True) # task=task)
    env = observation_dictionary_to_array_wrapper.ObservationDictionaryToArrayWrapper(env)
    env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(env,
        trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(action_limit=6.28318548203))

    action_low, action_high = env.action_space.low, env.action_space.high
    action_median = (action_low + action_high) / 2.
    dim_action = action_low.shape[0]
    action_selector_ids = []
    for dim in range(dim_action):
        action_selector_id = p.addUserDebugParameter(paramName='dim{}'.format(dim),
                                                    rangeMin=action_low[dim],
                                                    rangeMax=action_high[dim],
                                                    startValue=action_median[dim])
        action_selector_ids.append(action_selector_id)

    if FLAGS.video_dir:
        log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, FLAGS.video_dir)

    for _ in tqdm(range(800)):
        action = np.zeros(dim_action)
        for dim in range(dim_action):
            action[dim] = env.pybullet_client.readUserDebugParameter(
            action_selector_ids[dim])
        env.step(action)

    if FLAGS.video_dir:
        p.stopStateLogging(log_id)

if __name__ == "__main__":
  app.run(main)
