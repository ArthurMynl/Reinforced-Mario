from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
import gym
import numpy as np
import random
from tqdm import tqdm
import pickle
from QLearningAgent import QLearningAgent
import time 


def load_q_table(filename):
    with open(filename, 'rb') as file:
        q_table = pickle.load(file)
    return q_table


# Load the Q-table from file
q_table = load_q_table('q_table_checkpoint.pkl')

env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode='human')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

# Initialize the agent with the loaded Q-table
agent = QLearningAgent(env.action_space, q_table)

# Main game loop
frame_time = 1.0 / 60  # Time for each frame at 60 FPS

for episode in tqdm(range(10)):
    state = env.reset()
    state = agent.get_state(state)
    done = False

    while not done:
        start_time = time.time()  # Get the start time of the iteration

        action = agent.act(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = agent.get_state(next_state)
        done = terminated or truncated
        state = next_state

        elapsed_time = time.time() - start_time  # Time taken for the iteration
        sleep_time = frame_time - elapsed_time  # Calculate remaining time

        if sleep_time > 0:
            time.sleep(sleep_time)  # Sleep to maintain the frame rate

env.close()