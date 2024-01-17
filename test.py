from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym

env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = True
env.reset()
for step in range(100000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step([5])
    done = terminated or truncated

    if done:
       state = env.reset()

env.close()
