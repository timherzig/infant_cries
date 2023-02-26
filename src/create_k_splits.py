import os
import pandas as pd

pd.set_option("display.max_rows", None)

splits = 51
root = os.path.join('BabyCry2', str(splits) + '_fold_split')

train_df = pd.read_csv('BabyCry2/train.csv')
train_df = train_df.sample(frac=1).reset_index(drop=True)

ids = train_df['id'].value_counts().index.tolist()

val_len = len(ids)//splits
print(val_len)

for split in range(splits):
    # val_ids = ids[split*val_len:(split+1)*val_len]
    
    # val = train_df.loc[train_df['id'].isin(val_ids)]
    # train = train_df.loc[~train_df['id'].isin(val_ids)]
    val = train_df.loc[split*val_len:(split+1)*val_len]
    train = pd.concat([val, train_df]).drop_duplicates(keep=False)

    save_loc = os.path.join(root, str(split))
    os.makedirs(save_loc)

    val.to_csv(os.path.join(save_loc, 'val.csv'), index=False)
    train.to_csv(os.path.join(save_loc, 'train.csv'), index=False)






