import multiprocessing
from multiprocessing import Process, Pipe
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import numpy as np
import random
import pickle
from tqdm import tqdm



# Assuming your Q-table is a dictionary in the agent
def save_q_table(agent, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dict(agent.q_table), file)


def worker(remote, parent_remote, env_fn):
    try:
        parent_remote.close()
        env = env_fn()
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                next_state, reward, terminated, truncated, info = env.step(
                    data)  # Unpack all five values
                remote.send((next_state, reward, terminated, truncated, info))
            elif cmd == 'sample_action':
                remote.send(env.action_space.sample())
            elif cmd == 'reset':
                obs = env.reset()
                remote.send(obs)
            elif cmd == 'close':
                remote.close()
                break
            else:
                raise NotImplementedError
    except Exception as e:
        print("Error in worker process:", e)
        remote.send(None)  # Signal the main process about the error


class QLearningAgent:
    def __init__(self, action_space, shared_q_table):
        self.action_space = action_space
        self.q_table = shared_q_table  # Use the shared Q-table
        self.alpha = 0.1   # learning rate
        self.gamma = 0.6   # discount factor
        self.epsilon = 0.1  # exploration rate

    def get_state(self, observation):
        # Simplify the observation to a state
        # This can be a complex function depending on the observation space
        # For now, let's just convert it to a string as a placeholder
        return str(observation)

    def act(self, state):
        if random.uniform(0, 1) < self.epsilon:
            # Explore: choose a random action
            return self.action_space.sample()
        else:
            # Exploit: choose the best action based on the current Q-table
            return np.argmax(self.q_table.get(state, np.zeros(self.action_space.n)))

    def learn(self, state, action, reward, next_state, done):
        # Update Q-table using the Q-learning algorithm
        old_value = self.q_table.get(
            state, np.zeros(self.action_space.n))[action]
        next_max = np.max(self.q_table.get(
            next_state, np.zeros(self.action_space.n)))

        # Q-learning formula
        new_value = (1 - self.alpha) * old_value + self.alpha * \
            (reward + self.gamma * next_max)

        # Update the Q-table
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space.n)
        self.q_table[state][action] = new_value


class VecEnv():
    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, env_fn))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def sample_actions(self):
        return [remote.send(('sample_action', None)) or remote.recv() for remote in self.remotes]

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False

        # Check for None values
        if any(result is None for result in results):
            print("Error detected in one of the worker processes")
            # Handle the error appropriately, e.g., by stopping the training
            # For now, let's just raise an exception
            raise RuntimeError("Error detected in one of the worker processes")

        next_states, rewards, terminateds, truncateds, infos = zip(*results)
        return np.stack(next_states), np.stack(rewards), np.array(terminateds), np.array(truncateds), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


def create_env():
    return JoypadSpace(gym.make('SuperMarioBros-v2', apply_api_compatibility=True), SIMPLE_MOVEMENT)


# Main process
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    # Initialize a Manager for shared Q-table
    manager = multiprocessing.Manager()
    shared_q_table = manager.dict()

    # Initialize the Q-learning agent with the shared Q-table
    # Note: You need to define or get a sample action space for this initialization
    sample_env = create_env()  # Create a sample environment to get the action space
    agent = QLearningAgent(
        action_space=sample_env.action_space, shared_q_table=shared_q_table)
    sample_env.close()  # Close the sample environment

    # Initialize the vectorized environments
    num_envs = 10  # Example number of environments
    env_fns = [create_env for _ in range(num_envs)]
    vec_env = VecEnv(env_fns)

    # Loop over episodes
    for episode in tqdm(range(1000000)):
        states = vec_env.reset()
        done = [False] * num_envs

        while not all(done):
            # Get actions for all environments
            actions = [agent.act(agent.get_state(state)) for state in states]

            # Step in all environments
            vec_env.step_async(actions)
            next_states, rewards, terminateds, truncateds, infos = vec_env.step_wait()

            # Update Q-table based on experiences
            for i in range(num_envs):
                agent.learn(agent.get_state(states[i]), actions[i], rewards[i], agent.get_state(
                    next_states[i]), terminateds[i] or truncateds[i])

            states = next_states

    save_q_table(agent, 'q_table_checkpoint.pkl')
    vec_env.close()
