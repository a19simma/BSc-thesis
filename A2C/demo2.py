import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vizdoomEnv import VizDoomTrain
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import fnmatch
import os
import time
import numpy as np

SCENARIO = 'deadly_corridor'
MODEL_NAME = 'best_model_500000'

env = VizDoomTrain(SCENARIO)
env = Monitor(env)

model = A2C.load(MODEL_NAME)
for episode in range(2300):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        obs = np.array(obs)
        obs = obs.reshape((-1,) + env.observation_space.shape)
        action = model.predict(obs)[0]
        obs, reward, done, info = env.step(action[0])
        time.sleep(0.04)
        total_reward += reward
    print("episode finished with a score of: " + str(total_reward))
