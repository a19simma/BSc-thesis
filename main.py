from vizdoomEnv import VizDoomTrain
from callback import TrainCallback
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import optuna

# Basics methods for the vizdoom environment are:
# make_action which takes a list of button states given by an array of 0 or 1 with the 
# length of the number of buttons.

SCENARIO = 'deadly_corridor'
LOG_DIR = 'logs/' + SCENARIO
TOTAL_TIMESTEPS = 3e5

#Defaults are taken from the 2017 paper by schulman et al. https://arxiv.org/abs/1707.06347
def optimize_ppo(trial):
    return{
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-7,1e-1),
        'buffer_size':  int(1e5), # size of the buffer was reduced because of ram limitations.
        'learning_starts': TOTAL_TIMESTEPS/20,
        'batch_size':trial.suggest_loguniform('batch_size', 8,128),
        'tau': 1.0,
        'gamma': 0.99,
        'train_freq': trial.suggest_loguniform('train_freq', 1,50),
        'gradient_steps': trial.suggest_loguniform('gradient_steps', -1,10),
    }
def optimize_agent(trial):
    model_params = optimize_ppo(trial) 
    RUN_NAME = 'Trial_' + str(trial.number) + '_'
    for key in model_params:
        RUN_NAME += key + '=' + str(model_params[key]) + '_'
        RUN_NAME = RUN_NAME[:-1]   

    env = VizDoomTrain(SCENARIO)
    env = Monitor(env)
    model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, n_steps=2048, verbose=0, **model_params)
    logger = configure(LOG_DIR + '/' + RUN_NAME, ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)
    callback = TrainCallback(10000, LOG_DIR + '/' + RUN_NAME)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, log_interval=1) #decrease frequency of output with log_interval
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
    return mean_reward

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    try:
        study.optimize(optimize_agent, n_trials=40, gc_after_trial=True, n_jobs=-1)
    except KeyboardInterrupt:
        print('Interrupted by keyboard')