from nes_py.wrappers import JoypadSpace
import gym
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation, NormalizeObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import os

JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

FRAME_STACK_SIZE = 4
ENV_ID = "SuperMarioBros-v0"
NUM_PROCESSES = 8
CHECKPOINT_DIR = "./train"
LOG_DIR = "./logs"
RECORD_DIR = "./videos"


def make_env(ENV_ID: str, rank: int, seed: int = None):
    def _init():
        env = gym.make(ENV_ID, apply_api_compatibility=True)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = GrayScaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, shape=(84, 84))  # Ensure consistent shape
        env = NormalizeObservation(env)
        return env
    return _init

def make_eval_env(ENV_ID: str):
    def _init():
        env = gym.make(ENV_ID, apply_api_compatibility=True)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = GrayScaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, shape=(84, 84))  # Ensure consistent shape
        env = NormalizeObservation(env)
        return env
    return _init


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

# Create the evaluation environment
eval_env = DummyVecEnv([make_eval_env(ENV_ID)])
eval_env = VecFrameStack(eval_env, n_stack=FRAME_STACK_SIZE, channels_order='last')

# Setup the callback for training
callback = TrainAndLoggingCallback(check_freq=10000 / NUM_PROCESSES, save_path=CHECKPOINT_DIR)

# Setup the evaluation callback
eval_callback = EvalCallback(
    eval_env=eval_env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    eval_freq=10000 / NUM_PROCESSES,
    best_model_save_path=CHECKPOINT_DIR,
    log_path=LOG_DIR,
    deterministic=True,
    render=False
)

# object pour changer le learning rate du model qui va être chargé
custom_objects = { 'n_steps': 4096 }

# Charger le modèle précédemment enregistré
#model = PPO.load('./saved_models/Mario-1M', custom_objects=custom_objects)
model = PPO.load('./saved_models/Mario-1M')

model.set_env(env)

# Train the model
model.learn(total_timesteps=1000000, callback=[callback, eval_callback])

# Save the model
model_name = "Mario-1M"
model.save(f"./saved_models/{model_name}")