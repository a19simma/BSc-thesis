import pandas as pd
from matplotlib import axes, pyplot as plt
import seaborn as sns
sns.set(style="darkgrid", palette="muted", color_codes=True)

df_a2c_default_1 = pd.read_csv("default_1.csv", usecols=['rollout/ep_rew_mean', 'time/total_timesteps'], sep=',')
df_a2c_default_2 = pd.read_csv("default_2.csv", usecols=['rollout/ep_rew_mean', 'time/total_timesteps'], sep=',')
df_a2c_default_3 = pd.read_csv("default_3.csv", usecols=['rollout/ep_rew_mean', 'time/total_timesteps'], sep=',')

df_a2c_default_1 = df_a2c_default_1[::10000]
df_a2c_default_2 = df_a2c_default_2[::10000]
df_a2c_default_3 = df_a2c_default_3[::10000]

df_a2c_default_merged = pd.concat([df_a2c_default_1,df_a2c_default_2,df_a2c_default_3], ignore_index=True)

plt.figure(dpi=300)
sns.lineplot(x='time/total_timesteps', y="rollout/ep_rew_mean", label="A2C Default",data=df_a2c_default_merged, linewidth=1, color="red", alpha=0.5)
font = {'family' : 'DejaVu Sans',
        'size'   : 14}

plt.rc('font', **font)
plt.xlabel("Timesteps", fontsize=14)
plt.ylabel("Reward", fontsize=14)
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig('a2c_default_300_dpi.png')