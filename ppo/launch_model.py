import os
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

MODEL_NUMBER = 1000000

FRAME_STACK_SIZE = 4

env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
# Simplify controls
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# Grayscale observation
env = GrayScaleObservation(env, keep_dim=True)
# Wrap inside the dummy vector environment
env = DummyVecEnv([lambda: env])
# Stack 4 frames
env = VecFrameStack(env, n_stack=FRAME_STACK_SIZE, channels_order='last')

model = PPO.load(os.path.join("./ppo/train_part1", f"model_{MODEL_NUMBER}"), env=env)

# start the game with the trained model
state = env.reset()
for i in range(10000):
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()
env.close()