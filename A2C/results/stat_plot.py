import pandas as pd
from matplotlib import axes, pyplot as plt
import seaborn as sns
sns.set(style="darkgrid", palette="muted", color_codes=True)

df_a2c_default_1 = pd.read_csv("progress_best_trial.csv", usecols=['rollout/ep_rew_mean', 'time/total_timesteps'], sep=',')
#df_dqn_default_2 = pd.read_csv("default_2.csv", usecols=['rollout/ep_rew_mean', 'time/episodes', 'time/total_timesteps'], sep=',')
#df_dqn_default_3 = pd.read_csv("default_3.csv", usecols=['rollout/ep_rew_mean', 'time/episodes', 'time/total_timesteps'], sep=',')



#df_dqn_default_merged = pd.concat([df_dqn_default_1,df_dqn_default_2,df_dqn_default_3], ignore_index=True)

sns.lineplot(x='time/total_timesteps', y="rollout/ep_rew_mean", label="A2C best trial",data=df_a2c_default_1, linewidth=1, color="red", alpha=0.5)
font = {'family' : 'DejaVu Sans',
        'size'   : 14}

plt.rc('font', **font)
plt.xlabel("Timesteps")
plt.ylabel("Reward")
#plt.xlim(0,8.2e+3)
plt.show()