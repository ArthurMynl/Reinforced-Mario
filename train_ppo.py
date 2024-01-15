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
ENV_ID = "SuperMarioBros-v0"
NUM_PROCESSES = 8
CHECKPOINT_DIR = "./train"
LOG_DIR = "./logs"
RECORD_DIR = "./videos"


def make_env(ENV_ID: str, rank: int, seed: int = None):
    """
    Utility function for multiprocessed env.

    :param ENV_ID: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = gym.make(ENV_ID, apply_api_compatibility=True)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = GrayScaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, shape=84)
        env = NormalizeObservation(env)

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