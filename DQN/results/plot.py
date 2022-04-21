import pathlib
import pandas as pd
from matplotlib import axes, pyplot as plt
import seaborn as sns
sns.set(style="darkgrid", palette="muted", color_codes=True)

df_dqn_default_1 = pd.read_csv("default_1.csv", usecols=['rollout/ep_rew_mean', 'time/episodes', 'time/total_timesteps'], sep=',')
df_dqn_default_2 = pd.read_csv("default_2.csv", usecols=['rollout/ep_rew_mean', 'time/episodes', 'time/total_timesteps'], sep=',')
df_dqn_default_3 = pd.read_csv("default_3.csv", usecols=['rollout/ep_rew_mean', 'time/episodes', 'time/total_timesteps'], sep=',')
df_dqn_optimized =  pd.read_csv("optimized_1.csv", usecols=['rollout/ep_rew_mean', 'time/episodes', 'time/total_timesteps'], sep=',')

df_dqn_default_merged = pd.concat([df_dqn_default_1,df_dqn_default_2,df_dqn_default_3], ignore_index=True)
df_dqn_default_merged.to_csv(str(pathlib.Path(__file__).parent.resolve()) + "\\merged.csv", index=False)
#df_dqn_default_merged['mean'] = df_dqn_default_merged.groupby('time/episodes').mean()
#df_dqn_default_merged['mov_avg'] = df_dqn_default_merged['mean'].rolling(10).mean()


#sns.lineplot(x="time/total_timesteps", y="rollout/ep_rew_mean", label="A2C Learning Rate 0.00007", data=df_a2c, linewidth=1, color="green")
#sns.lineplot(x="time/total_timesteps", y="mov_avg", data=df_a2c, label="Learning Rate 0.00007", linewidth=1, color="green")
#sns.lineplot(x="time/total_timesteps", y="mov_avg", data=df_dqn, label="Learning Rate 0.0001", linewidth=1, color="purple")
sns.lineplot(x='time/episodes', y="rollout/ep_rew_mean", label="DQN default",data=df_dqn_default_merged, linewidth=2, color="blue", alpha=0.5)
#sns.lineplot(x="time/total_timesteps", y="mov_avg", data=df_dqn_default_merged, marker='o', markevery=25, linewidth=2, color="blue", alpha=0.5)
sns.lineplot(x='time/episodes', y="rollout/ep_rew_mean", label="DQN optimized", data=df_dqn_optimized, linewidth=2, color="green", alpha=0.5)

plt.xlabel("Timesteps")
plt.ylabel("Reward")
plt.show()