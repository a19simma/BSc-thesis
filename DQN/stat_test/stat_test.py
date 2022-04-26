import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from scipy.stats import mannwhitneyu
from vizdoomEnv import VizDoomTrain
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

env = VizDoomTrain('deadly_corridor')
env = Monitor(env)
model_dir = os.path.dirname(os.path.abspath(__file__))
optimized_model = DQN.load(model_dir + "\optimized_model.zip")
default_model = DQN.load(model_dir + "\default_model.zip")

if not os.path.exists(model_dir + '\optimized_evaluation_sample.csv'):
    data_optimized = []    
    for n in range(50):
        mean_reward, _ = evaluate_policy(optimized_model, env, n_eval_episodes=1)
        data_optimized.append(mean_reward)
    data_optimized = np.array(data_optimized) 
    np.savetxt('optimized_evaluation_sample.csv', data_optimized)
else: 
    data_optimized = np.loadtxt('optimized_evaluation_sample.csv')

if not os.path.exists(model_dir + '\default_evaluation_sample.csv'):
    data_default = []   
    for n in range(50):
        mean_reward, _ = evaluate_policy(default_model, env, n_eval_episodes=1)
        data_default.append(mean_reward)
    data_default = np.array(data_default) 
    np.savetxt('default_evaluation_sample.csv', data_optimized)
else: 
    data_default = np.loadtxt('default_evaluation_sample.csv')

statistic, pvalue = mannwhitneyu(data_default,data_optimized)

print("p-value " + str(pvalue))