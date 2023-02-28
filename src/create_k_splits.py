import os
import pandas as pd

pd.set_option("display.max_rows", None)

ids_j = ['J30', 'J01', 'J21', 'J14', 'J18', 'J19', 'J27', 'J26', 'J15', 'J03', 'J12', 'J02', 'J17', 'J29', 'J08', 'J11', 'J04', 'J06', 'J20', 'J10', 'J05', 'J22', 'J16', 'J25', 'J07']
ids_g = ['G02', 'G17', 'G10', 'G16', 'G25', 'G14', 'G27', 'G07', 'G30', 'G28', 'G09', 'G03', 'G06', 'G32', 'G18', 'G19', 'G12', 'G24', 'G20', 'G33', 'G15', 'G08', 'G22', 'G05', 'G23']

splits = len(ids_g)

root = os.path.join('BabyCry4-1', str(splits) + '_fold_split')

train_df = pd.read_csv('BabyCry4-1/train.csv')
train_df = train_df.sample(frac=1).reset_index(drop=True)


# ids = train_df['id'].value_counts().index.tolist()

# val_len = len(ids)//splits
# print(val_len)
# print(train_df['id'].unique())


# for split in range(splits):
#     # val_ids = ids[split*val_len:(split+1)*val_len]
    
#     # val = train_df.loc[train_df['id'].isin(val_ids)]
#     # train = train_df.loc[~train_df['id'].isin(val_ids)]
    
#     # val = train_df.loc[split*val_len:(split+1)*val_len]
#     # train = pd.concat([val, train_df]).drop_duplicates(keep=False)
#     g = ids_g.pop()
#     j = ids_j.pop()

#     val = train_df.loc[(train_df['id'] == g) | (train_df['id'] == j)]
#     train = train_df.loc[(train_df['id'] != g) & (train_df['id'] != j)]

#     save_loc = os.path.join(root, str(split))
#     os.makedirs(save_loc)

#     val.to_csv(os.path.join(save_loc, 'val.csv'), index=False)
#     train.to_csv(os.path.join(save_loc, 'train.csv'), index=False)






