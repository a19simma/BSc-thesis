import pandas as pd

df_a2c_optimized_evaluation = pd.read_csv("a2c_optimized_evaluation_sample.csv", sep=';')

print(df_a2c_optimized_evaluation.mean())