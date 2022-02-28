import optuna
import gym
import numpy as np
import sys

from vizdoomEnv import VizDoomTrain
from callback import TrainCallback
from matplotlib import pyplot as plt
from stable_baselines3.common import env_checker
from stable_baselines3 import PPO

from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.evaluation import evaluate_policy #Använder inte dessa egentligen
from stable_baselines3.common.env_util import make_vec_env #  cmd_util har döpts om till env_util, men har deprekerats och kommer tas bort i framtiden

sys.setrecursionlimit(800)


# Basics methods for the vizdoom environment are:
# make_action which takes a list of button states given by an array of 0 or 1 with the 
# length of the number of buttons.
#

LOG_DIR = '../vizdoomlog/logs'



def optimize_ppo(trial):
    return{
        #'n_steps': int(trial.suggest_loguniform('n_steps', 16, 2048)),                     Fungerar
        'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),#                           Fungerar
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),#              Fungerar
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),#                      Fungerar
        #'batch_size' : int(trial.suggest_loguniform('batch_size', 1, n_steps * n_envs)),   Fungerar, men vi behöver den nog inte
        'n_epochs': int(trial.suggest_loguniform('n_epochs', 1, 48))#,                      Fungerar
        #'cliprange': trial.suggest_uniform('cliprange', 0.1, 0.4),                         Fungerar inte
        #'noptepochs': int(trial.suggest_loguniform('noptepochs', 1, 48)),                  Fungerar inte
        #'lam': trial.suggest_uniform('lam', 0.8, 1.)                                       Fungerar inte
    }


def optimize_agent(trial):

    model_params = optimize_ppo(trial) #Den kallade till agentens optimering innan, som orsakade rekursivt problem

    env = VizDoomTrain('defend_the_center')
    callback = TrainCallback(10000) #1h 18min ifall en save path vill läggas till
    model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, n_steps=2048, verbose=1, **model_params)
    #model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=4096)
    
    
    #n_steps ökas vid mer komplexa miljöer
    model.learn(total_timesteps=70000, callback=callback)
    #model.learn(total_timesteps=10000, callback=callback)

#Evaluation environment is not wrapped with a ''Monitor''  wrapper, med det nedan får vi inget error men vi får inte heller hela den utskrift vi vill ha.
    monitor_env = Monitor(env)
    mean_reward, _ = evaluate_policy(model, monitor_env, n_eval_episodes=10) 
    return -1 * mean_reward


if __name__ == '__main__':
    study = optuna.create_study(study_name='FirstMassOPT')
    try:
        study.optimize(optimize_agent, n_trials=100, gc_after_trial=True)
    except KeyboardInterrupt:
        print('Interrupted by keyboard')

        