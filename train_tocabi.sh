#!/usr/bin/env bash

#./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task Isaac-Tocabi-AMP-Walk-Direct-v0 # --headless 
#./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Tocabi-AMP-Walk-Direct-v0 --algorithm AMP --max_iterations 5000 --headless

#./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --play Isaac-Velocity-Flat-Tocabi-Direct-v0 --num_envs 1 
#./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-Tocabi-Direct-v0 --headless 


./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Velocity-Flat-Tocabi-Direct-Play-v0 --num_envs 1 --checkpoint /home/jhk/IsaacSim/IsaacLab/logs/rsl_rl/tocabi_flat_direct/2025-09-02_19-34-11/model_4999.pt 


#./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py --task Isaac-Tocabi-AMP-Walk-Direct-v0  --algorithm AMP --num_envs 1 --checkpoint /home/jhk/IsaacSim/IsaacLab/logs/skrl/tocabi_amp_walk/2025-08-23_17-53-26_amp_torch/checkpoints/best_agent.pt 
