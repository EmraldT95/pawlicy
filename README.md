# Pawlicy - Learning to walk RL project

### Training

*Required python version: >=3.7. Also ensure all the packages mentioned in `requirements.txt` are installed.*

To train the robot, run the following in the terminal:
```bash
python -m pawlicy.run -m train -a PPO
```

The available command line flags are
<pre>
  --mode, -m                    Allowed values: train, test<em>(Default)</em>
  --algorithm, -a               Allowed values: PPO, TD3, SAC<em>(Default)</em>
  --motor_control_mode, -mcm    Allowed values: Position, Torque<em>(Default)</em>
  --visualize, -v               If set, renders the 3D simulation.
  --randomise_terrain, -rt      Allowed values: True, False<em>(Default)</em>
  --total_timesteps, -tts       Allowed values: 1000000<em>(Default)</em>, can be any integer value.
  --path, -p                    Path to save to model to or load the model from.
  --max_episode_steps, -mes     Allowed values: 1000<em>(Default)</em>, can be any integer value.
  --total_num_eps, -tne         Allowed values: 20<em>(Default)</em>, can be any integer value
                                  and is used only in testing.
  --load_final_model, -lfm      If set, loads the final model instead of the best model.
</pre>
