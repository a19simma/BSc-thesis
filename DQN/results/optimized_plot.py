import pathlib
import pandas as pd
from matplotlib import axes, pyplot as plt
import seaborn as sns
sns.set(style="darkgrid", palette="muted", color_codes=True)

df_dqn_optimized_2 = pd.read_csv("optimized_2.csv", usecols=['rollout/ep_rew_mean', 'time/episodes', 'time/total_timesteps'], sep=',')
df_dqn_optimized_3 = pd.read_csv("optimized_3.csv", usecols=['rollout/ep_rew_mean', 'time/episodes', 'time/total_timesteps'], sep=',')
df_dqn_optimized_1 = pd.read_csv("optimized_1.csv", usecols=['rollout/ep_rew_mean', 'time/episodes', 'time/total_timesteps'], sep=',')
df_dqn_optimized_4 = pd.read_csv("optimized_4.csv", usecols=['rollout/ep_rew_mean', 'time/episodes', 'time/total_timesteps'], sep=',')
df_dqn_optimized_5 = pd.read_csv("optimized_5.csv", usecols=['rollout/ep_rew_mean', 'time/episodes', 'time/total_timesteps'], sep=',')

df_dqn_optimized_merged = pd.concat([df_dqn_optimized_1,df_dqn_optimized_3,df_dqn_optimized_4,df_dqn_optimized_5], ignore_index=True)

sns.lineplot(x='time/episodes', y="rollout/ep_rew_mean", label="DQN optimized",data=df_dqn_optimized_merged, linewidth=1, color="blue", alpha=0.5)
font = {'family' : 'DejaVu Sans',
        'size'   : 14}

plt.rc('font', **font)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.xlim(0,1.4e+4)
plt.show()