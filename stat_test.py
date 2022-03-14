from scipy import stats
import pandas as pd

df_a2c = pd.read_csv("logs/progress_a2c.csv", usecols=['rollout/ep_rew_mean', 'time/total_timesteps'], sep=',')
df_dqn = pd.read_csv("logs/progress_dqn.csv", usecols=['rollout/ep_rew_mean', 'time/total_timesteps'], sep=',')
df_ppo = pd.read_csv("logs/progress_ppo.csv", usecols=['rollout/ep_rew_mean', 'time/total_timesteps'], sep=',')

p_adam_vs_rmsprop = stats.ttest_ind(
    df_a2c["rollout/ep_rew_mean"],
    df_dqn["rollout/ep_rew_mean"]
)
print(p_adam_vs_rmsprop)