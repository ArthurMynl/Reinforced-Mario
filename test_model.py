from nes_py.wrappers import JoypadSpace
import gym
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation, NormalizeObservation, RecordVideo
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from matplotlib import pyplot as plt
import os

FRAME_STACK_SIZE = 4
SAVED_MODELS_DIR = "./saved_models"
CHECKPOINT_DIR = "./train"

# Base environment
env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
# Simplify controls
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# Grayscale observation
env = GrayScaleObservation(env, keep_dim=True)
# Resize to a square image
env = ResizeObservation(env, shape=84)
# Normalize the observation
env = NormalizeObservation(env)
# Wrap inside the dummy vector environment
env = DummyVecEnv([lambda: env])
# Stack 4 frames
env = VecFrameStack(env, n_stack=FRAME_STACK_SIZE, channels_order='last')


model = PPO.load(os.path.join(SAVED_MODELS_DIR, "Mario-3M"), env=env)
# model = PPO.load(os.path.join(CHECKPOINT_DIR, "model_200000"), env=env)


# start the game with the trained model
state = env.reset()
for i in range(10000):
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()
env.close()