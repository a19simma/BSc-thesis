import pandas as pd
from matplotlib import axes, pyplot as plt
import seaborn as sns
sns.set(style="darkgrid", palette="muted", color_codes=True)

df_a2c = pd.read_csv("progress_LR00007.csv", usecols=['rollout/ep_rew_mean', 'time/total_timesteps'], sep=',')
df_dqn = pd.read_csv("DQN_3e6_T_RewardShaping.csv", usecols=['rollout/ep_rew_mean', 'time/total_timesteps'], sep=',')
df_ppo = pd.read_csv("progress_ppo.csv", usecols=['rollout/ep_rew_mean', 'time/total_timesteps'], sep=',')

df_a2c['mov_avg'] = df_a2c['rollout/ep_rew_mean'].rolling(20).mean()
df_dqn['mov_avg'] = df_dqn['rollout/ep_rew_mean'].rolling(20).mean()
df_ppo['mov_avg'] = df_ppo['rollout/ep_rew_mean'].rolling(20).mean()

sns.lineplot(x="time/total_timesteps", y="rollout/ep_rew_mean", label="A2C Learning Rate 0.00007", data=df_a2c, linewidth=1, color="green")
#sns.lineplot(x="time/total_timesteps", y="mov_avg", data=df_a2c, label="Learning Rate 0.00007", linewidth=1, color="green")
sns.lineplot(x="time/total_timesteps", y="rollout/ep_rew_mean", label="DQN Learning Rate 0.0001", data=df_dqn, linewidth=1, color="purple")
#sns.lineplot(x="time/total_timesteps", y="mov_avg", data=df_dqn, label="Learning Rate 0.0001", linewidth=1, color="purple")
sns.lineplot(x="time/total_timesteps", y="rollout/ep_rew_mean", label="PPO Learning Rate 0.0001",data=df_ppo, linewidth=1, color="blue")
#sns.lineplot(x="time/total_timesteps", y="mov_avg", data=df_ppo, label="PPO Learning Rate 0.0001", linewidth=1, color="blue")

plt.xlabel("Timesteps")
plt.ylabel("Reward")
plt.show()