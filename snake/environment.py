from gym_snake.envs.snake_env import SnakeEnv
from gym import GoalEnv
from gym import spaces
import numpy as np


class TrainShapedRewardEnv(SnakeEnv):

    def __init__(self, max_steps=None, shaped_reward=True, **kwargs):
        SnakeEnv.__init__(self, **kwargs)

        self.shaped_reward = shaped_reward

        if max_steps is None:
            self.max_steps = (self.grid_size[1] + self.grid_size[0]) * 10
        else:
            self.max_steps = max_steps

    def distance(self):
        head = self.controller.snakes[0].head
        last_food = self.controller.grid.last_food
        dist = np.abs(head-last_food).sum()
        return dist

    def reset(self):
        obs = SnakeEnv.reset(self)
        self.current_step = 0
        self.prev_dist = self.distance()
        return obs

    def step(self, action):
        obs, rewards, done, info = SnakeEnv.step(self, action)

        if rewards > 0:
            self.prev_dist = self.distance()

        if self.shaped_reward and self.controller.snakes[0]:
            dist = self.distance()
            dist_change = self.prev_dist - dist
            rewards += dist_change*1e-2
            self.prev_dist = dist

        self.current_step += 1
        done = done or self.current_step >= self.max_steps
        return obs, rewards, done, info
