# installing dependencies

requires python 3.7-3.9

pip install vizdoom

pip install gym

pip install stable-baselines3[extra]

run main.py

run tensorboard --logdir='LOGDIR/PPO_{n}' in commandline for
the tensorboard interface. The logdir path is set by the LOG_DIR variable
