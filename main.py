import optuna
import sys
from matplotlib import pyplot as plt
from stable_baselines3.common import env_checker
from vizdoomEnv import VizDoomTrain
from callback import TrainCallback
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy 


sys.setrecursionlimit(800)


SCENARIO = 'deadly_corridor'
LOG_DIR = 'logs/' + SCENARIO + '_PPO_TEST'



def optimize_ppo(trial):
    return{
        #'n_steps': int(trial.suggest_loguniform('n_steps', 16, 2048)),                     Fungerar
        #'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),#                           Fungerar
        #'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),#              Fungerar
        #'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),#                      Fungerar
        #'batch_size' : int(trial.suggest_loguniform('batch_size', 1, n_steps * n_envs)),   Fungerar, men vi behöver den nog inte
        #'n_epochs': int(trial.suggest_loguniform('n_epochs', 1, 48))#,                      Fungerar
        #'cliprange': trial.suggest_uniform('cliprange', 0.1, 0.4),                         Fungerar inte
        #'noptepochs': int(trial.suggest_loguniform('noptepochs', 1, 48)),                  Fungerar inte
        #'lam': trial.suggest_uniform('lam', 0.8, 1.)                                       Fungerar inte
    }


def optimize_agent(trial):

    model_params = optimize_ppo(trial) 

    RUN_NAME = ''
    for key in model_params:
        RUN_NAME += key + '=' + str(model_params[key]) + '_'
    RUN_NAME = RUN_NAME[:-1]

    env = VizDoomTrain(SCENARIO)
    env = Monitor(env, (LOG_DIR + '/' + RUN_NAME))
    #model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, n_steps=2048, verbose=1, **model_params)
    #model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, learning_rate=0.0001, n_steps=2048, verbose=1, **model_params)
    model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, learning_rate=0.0001, n_steps=2048, clip_range=.1, gamma=.95, gae_lambda=.9, verbose=1, **model_params)
    
    logger = configure(LOG_DIR + '/' + RUN_NAME, ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)
    callback = TrainCallback(50000, LOG_DIR + '/' + RUN_NAME)
    model.learn(total_timesteps=500000, callback=callback)

    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=3) 
    return mean_reward


if __name__ == '__main__':
    study = optuna.create_study()
    try:
        study.optimize(optimize_agent, n_trials=3, gc_after_trial=True)
    except KeyboardInterrupt:
        print('Interrupted by keyboard')

        