import os 
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from matplotlib import pyplot as plt
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT 
import gym

JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

env = gym.make('SuperMarioBros-1-3-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4,channels_order='last')

CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

model = PPO.load('./train/best_model_1000000')

state = env.reset()
while True: 

    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()