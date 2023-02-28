import os
import pandas as pd

root = 'BabyCry4-1'

test_df = pd.read_csv(os.path.join(root, 'test.csv'))
os.makedirs(os.path.join(root, 'test'), exist_ok=True)

for id in test_df['id'].unique():
    df = test_df.loc[test_df['id'] == id]
    # print(os.path.join(root, 'test', id + '.csv'))
    df.to_csv(os.path.join(root, 'test', id + '.csv'))