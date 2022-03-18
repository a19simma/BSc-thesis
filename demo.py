from vizdoomEnv import VizDoomTrain
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import time
import numpy as np
import tkinter
from tkinter import filedialog

env = VizDoomTrain('defend_the_center', visible=True)
env = Monitor(env)
tkinter.Tk().withdraw()
model_dir = filedialog.askopenfilename()

model = DQN.load(model_dir)
for episode in range(10):
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
