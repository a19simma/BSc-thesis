import pandas as pd
from matplotlib import axes, pyplot as plt
import seaborn as sns
sns.set(style="darkgrid", palette="muted", color_codes=True)

df_ppo_optimized_1 = pd.read_csv("optimized_1.csv", usecols=['rollout/ep_rew_mean', 'time/total_timesteps'], sep=',')
df_ppo_optimized_2 = pd.read_csv("optimized_2.csv", usecols=['rollout/ep_rew_mean', 'time/total_timesteps'], sep=',')
df_ppo_optimized_3 = pd.read_csv("optimized_3.csv", usecols=['rollout/ep_rew_mean', 'time/total_timesteps'], sep=',')

df_a2c_default_merged = pd.concat([df_ppo_optimized_1,df_ppo_optimized_2,df_ppo_optimized_3], ignore_index=True)

plt.figure(dpi=1200)
sns.lineplot(x='time/total_timesteps', y="rollout/ep_rew_mean", label="PPO Optimized",data=df_a2c_default_merged, linewidth=1, color="blue", alpha=0.5)
font = {'family' : 'DejaVu Sans',
        'size'   : 14}

plt.rc('font', **font)
plt.xlabel("Timesteps", fontsize=18)
plt.ylabel("Reward", fontsize=18)
plt.ylim([0, 1])
plt.savefig('ppo_optimized.png')