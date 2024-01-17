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
import cv2
import numpy as np
import os


FRAME_STACK_SIZE = 4
ENV_ID = "SuperMarioBros-v2"
NUM_PROCESSES = 8
CHECKPOINT_DIR = "./train"
LOG_DIR = "./logs"


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

def make_env(env_id: str, rank: int, seed: int = None):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = gym.make(env_id, apply_api_compatibility=True)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = ResizeObservation(env, shape=(144, 154))
        env = GrayScaleObservation(env, keep_dim=True)
        env = CannyFilterWrapper(env, low_threshold=50, high_threshold=100)
        return env
    return _init


# class to automatically save models while training
class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
    
    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            self.model.save(os.path.join(self.save_path, f"model_{self.n_calls * NUM_PROCESSES}"))
        
        return True
    

# Create n environments
env = DummyVecEnv([make_env(ENV_ID, i) for i in range(NUM_PROCESSES)])
# Stack 4 frames
env = VecFrameStack(env, n_stack=FRAME_STACK_SIZE, channels_order='last')

# setup the callback
callback = TrainAndLoggingCallback(check_freq=10000 / NUM_PROCESSES, save_path=CHECKPOINT_DIR)

# create the model
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, n_steps=512)

# train the model
model.learn(total_timesteps=1000000, callback=callback)

model_name = "Mario-1M"
model.save(f"./saved_models/{model_name}")