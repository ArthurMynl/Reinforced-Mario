import torch
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from agent import Agent
from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers
import os
from utils import *
import pyglet
from torch.utils.tensorboard import SummaryWriter

lib = pyglet.lib.load_library('GLU')
print("lib ", lib)

import ctypes.util
print(ctypes.util.find_library('GLU'))

JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

model_path = os.path.join("models", get_current_date_time_string())
os.makedirs(model_path, exist_ok=True)

if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
    print(torch.__version__)
    print(torch.version.cuda)
else:
    print("CUDA is not available")
    print(torch.__version__)
    print(torch.version.cuda)

ENV_NAME = 'SuperMarioBros-1-3-v0'
SHOULD_TRAIN = False
CKPT_SAVE_INTERVAL = 100
NUM_OF_EPISODES = 50_000

env = gym_super_mario_bros.make(ENV_NAME, render_mode='human', apply_api_compatibility=True)
#env = gym_super_mario_bros.make(ENV_NAME, apply_api_compatibility=True)

env = JoypadSpace(env, RIGHT_ONLY)
env = apply_wrappers(env)

agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

folder_name = "checkpoints"
ckpt_name = "model_15900_iter.pt"
agent.load_model(os.path.join("models", folder_name, ckpt_name))
agent.epsilon = 0.5185334470441866

if not SHOULD_TRAIN:
    folder_name = "checkpoints"
    ckpt_name = "model_15900_iter.pt"
    agent.load_model(os.path.join("models", folder_name, ckpt_name))
    agent.epsilon = 0.2
    agent.eps_min = 0.0
    agent.eps_decay = 0.0

# Créer un objet SummaryWriter, spécifiant le dossier où les logs seront sauvegardés
writer = SummaryWriter('logs')

env.reset()
next_state, reward, done, trunc, info = env.step(action=0)

for i in range(NUM_OF_EPISODES):    
    print("Episode:", i)
    done = False
    state, _ = env.reset()
    total_reward = 0
    while not done:
        a = agent.choose_action(state)
        new_state, reward, done, truncated, info  = env.step(a)
        total_reward += reward

        if SHOULD_TRAIN:
            agent.store_in_memory(state, a, reward, new_state, done)
            agent.learn()

        state = new_state

    # Enregistrez les informations pour TensorBoard
    writer.add_scalar('Total Reward', total_reward, i)
    writer.add_scalar('Epsilon', agent.epsilon, i)
    writer.add_scalar('Replay Buffer Size', len(agent.replay_buffer), i)
    writer.add_scalar('Learn Step Counter', agent.learn_step_counter, i)
    
    # Ajoutez d'autres paramètres que vous souhaitez surveiller
    # Exemples :
    # writer.add_scalar('Average Loss', your_loss_value, i)
    # writer.add_scalar('Average Episode Length', your_episode_length, i)
    print("Total reward:", total_reward, "Epsilon:", agent.epsilon, "Size of replay buffer:", len(agent.replay_buffer),
          "Learn step counter:", agent.learn_step_counter)
    if SHOULD_TRAIN and (i + 1) % CKPT_SAVE_INTERVAL == 0:
        agent.save_model(os.path.join(model_path, "model_" + str(i + 1) + "_iter.pt"))
    
    print("Total reward:", total_reward)


# Fermez le writer à la fin de l'entraînement
writer.close()

env.close()
