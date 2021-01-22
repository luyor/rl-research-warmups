# install environment:
# https://github.com/grantsrb/Gym-Snake#installation

import os

import gym
import gym_snake
from stable_baselines import HER, ACER, PPO2, DQN
from stable_baselines.common import make_vec_env
from stable_baselines.common.callbacks import (CheckpointCallback,
                                               EvalCallback,
                                               StopTrainingOnRewardThreshold)
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

from network import cnn
from environment import TrainShapedRewardEnv

directory = os.path.dirname(__file__)
os.chdir(directory)

if __name__ == '__main__':
    env_id = 'snake-v0'
    # env = TrainShapedRewardEnv()
    env = make_vec_env(TrainShapedRewardEnv, n_envs=8,
                       env_kwargs={'max_steps': 500, 'snake_size': 6})

    eval_env = TrainShapedRewardEnv(shaped_reward=False)

    checkpoint_callback = CheckpointCallback(
        save_freq=1000, save_path='./logs/')
    eval_callback = EvalCallback(
        eval_env, best_model_save_path='./logs', verbose=1)
    # model = ACER(CnnPolicy, env, policy_kwargs={'cnn_extractor': cnn},
    #              verbose=1, tensorboard_log='./tensorboard')
    model = ACER.load('saved_model', env, tensorboard_log='./tensorboard')

    # model = DQN('CnnPolicy', env, policy_kwargs={'cnn_extractor': cnn},
    #             verbose=1, tensorboard_log='./tensorboard')
    # model = DQN.load('saved_model', env, tensorboard_log='./tensorboard')
    model.learn(total_timesteps=1000000, callback=eval_callback,
                reset_num_timesteps=False)

    model.save('saved_model')
