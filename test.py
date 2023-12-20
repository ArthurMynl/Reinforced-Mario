from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
import gym
import numpy as np
import random
from tqdm import tqdm
import pickle
from QLearningAgent import QLearningAgent

# Assuming your Q-table is a dictionary in the agent
def save_q_table(agent, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dict(agent.q_table), file)


env = gym.make('SuperMarioBros-v3', apply_api_compatibility=True, render_mode='human')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

done = True
env.reset()

# Initialize the agent
# Define your bins here
cart_position_bins = np.linspace(-2.4, 2.4, 10)
cart_velocity_bins = np.linspace(-3, 3, 10)
pole_angle_bins = np.linspace(-0.2, 0.2, 10)
pole_velocity_bins = np.linspace(-3, 3, 10)

bins = [cart_position_bins, cart_velocity_bins, pole_angle_bins, pole_velocity_bins]

# Initialize your agent with these bins
# Assuming env is your environment instance
agent = QLearningAgent(env.action_space, bins=bins)

# Main game loop
for episode in tqdm(range(10000)):
    state = env.reset()
    state = agent.get_state(state)
    done = False

    while not done:
        # Select action based on the current state
        action = agent.act(state)

        # Take the action and observe the outcome
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = agent.get_state(next_state)
        done = terminated or truncated

        # Let the agent learn from the experience
        agent.learn(state, action, reward, next_state, done)

        # Transition to the next state
        state = next_state

    save_q_table(agent, 'q_table_checkpoint.pkl')
    # Optionally, decrease epsilon to reduce the amount of random actions
    agent.epsilon *= 0.99

env.close()