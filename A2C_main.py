from vizdoomEnv import VizDoomTrain
from callback import TrainCallback
from matplotlib import pyplot as plt
from stable_baselines3.common import env_checker
from stable_baselines3 import A2C
from packaging import version

LOG_DIR = '../vizdoomlog/logs'

env = VizDoomTrain('basic')

callback = TrainCallback(10000)

model = A2C('MlpPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=2048)

model.learn(total_timesteps=1000, callback=callback)