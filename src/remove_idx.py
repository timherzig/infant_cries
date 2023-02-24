import pandas as pd

df = pd.read_csv('BabyCry/train.csv', index_col=[0])
df.to_csv('BabyCry/train_fix.csv', index=False)