#!/usr/bin/env bash

# ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task Isaac-Tocabi-AMP-Walk-Direct-v0 --headless
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Tocabi-AMP-Walk-Direct-v0 --algorithm AMP --max_iterations 5000 --headless