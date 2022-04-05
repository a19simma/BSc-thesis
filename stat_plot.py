import pandas as pd
from matplotlib import axes, pyplot as plt
import seaborn as sns
sns.set(style="darkgrid", palette="muted", color_codes=True)

df_a2c = pd.read_csv("progress_a2c.csv", usecols=['rollout/ep_rew_mean', 'time/total_timesteps'], sep=',')
#df_dqn = pd.read_csv("progress_dqn.csv", usecols=['rollout/ep_rew_mean', 'time/total_timesteps'], sep=',')
#df_ppo = pd.read_csv("logs/progress_ppo.csv", usecols=['rollout/ep_rew_mean', 'time/total_timesteps'], sep=',')

df_a2c['mov_avg'] = df_a2c['rollout/ep_rew_mean'].rolling(20).mean()
#df_dqn['mov_avg'] = df_dqn['rollout/ep_rew_mean'].rolling(20).mean()
#df_ppo['mov_avg'] = df_ppo['rollout/ep_rew_mean'].rolling(20).mean()

sns.lineplot(x="time/total_timesteps", y="rollout/ep_rew_mean", data=df_a2c, linewidth=0.75, color="green", alpha=0.4)
sns.lineplot(x="time/total_timesteps", y="mov_avg", data=df_a2c, label="A2C mean", linewidth=0.75, color="green")
#sns.lineplot(x="time/total_timesteps", y="rollout/ep_rew_mean", data=df_dqn, linewidth=0.75, color="orange", alpha=0.4)
#sns.lineplot(x="time/total_timesteps", y="mov_avg", data=df_dqn, label="DQN mean", linewidth=0.75, color="orange")
#sns.lineplot(x="time/total_timesteps", y="rollout/ep_rew_mean", data=df_ppo, linewidth=0.75, color="blue", alpha=0.4)
#sns.lineplot(x="time/total_timesteps", y="mov_avg", data=df_ppo, label="PPO mean", linewidth=0.75, color="blue")

plt.xlabel("Timesteps")
plt.ylabel("Reward")
plt.show()