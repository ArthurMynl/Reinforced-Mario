import os
import cv2
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
import numpy as np
from stable_baselines3 import PPO
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from gym.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

MODEL_NUMBER = 2000000
env_id = 'SuperMarioBros-v2'

FRAME_STACK_SIZE = 4

class CannyFilterWrapper(gym.ObservationWrapper):
    def __init__(self, env, low_threshold=50, high_threshold=150):
        super().__init__(env)
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(env.observation_space.shape[0], env.observation_space.shape[1], 1), dtype=np.uint8)

    def observation(self, obs):
        # Appliquer le filtre de Canny
        canny = cv2.Canny(obs, self.low_threshold, self.high_threshold)
        # Ajouter une dimension supplémentaire pour correspondre à l'espace d'observation
        canny = np.expand_dims(canny, axis=-1)
        return canny

env = gym.make(env_id, apply_api_compatibility=True, render_mode='human')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = ResizeObservation(env, shape=(144, 154))
env = GrayScaleObservation(env, keep_dim=True)
env = CannyFilterWrapper(env, low_threshold=50, high_threshold=100)
# Wrap inside the dummy vector environment
env = DummyVecEnv([lambda: env])
# Stack 4 frames
env = VecFrameStack(env, n_stack=FRAME_STACK_SIZE, channels_order='last')

model = PPO.load(os.path.join("./ppo_2/train", f"model_{MODEL_NUMBER}"), env=env)

# start the game with the trained model
state = env.reset()
for i in range(10000):
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()
env.close()