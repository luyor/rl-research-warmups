import gym
import gym_snake
from stable_baselines import ACER, DQN
from stable_baselines.common import make_vec_env
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.evaluation import evaluate_policy

from environment import TrainShapedRewardEnv
from network import cnn

import os

directory = os.path.dirname(__file__)
os.chdir(directory)

env = TrainShapedRewardEnv(shaped_reward=False, snake_size=3)

model = ACER.load('saved_model')
# model = ACER.load('logs/best_model')
# model = DQN.load('best_model/best_model')
evaluate_policy(model, env, render=True)
