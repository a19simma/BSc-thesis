import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.stats import mannwhitneyu
from vizdoomEnv import VizDoomTrain
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

env = VizDoomTrain('deadly_corridor')
env = Monitor(env)
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_DQN = np.loadtxt(root_dir + '\DQN\stat_test\default_evaluation_sample.csv')
data_PPO = np.loadtxt(root_dir + '\PPO\stat_test\default_evaluation_sample.csv')
data_A2C = np.loadtxt(root_dir + '\A2C\stat_test\default_evaluation_sample.csv')

statistic, pvalue = mannwhitneyu(data_PPO,data_DQN)
print("p-value comparing the samples of PPO and DQN: " + str(pvalue))

statistic, pvalue = mannwhitneyu(data_PPO,data_A2C)
print("p-value comparing the samples of PPO and A2C: " + str(pvalue))

statistic, pvalue = mannwhitneyu(data_DQN, data_A2C)
print("p-value comparing the samples of DQN and A2C: " + str(pvalue))
