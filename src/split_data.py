import math
import random
import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from augment_data import augment_data

pd.set_option("display.max_rows", None)

df = pd.read_csv('BabyCry4-1/babycry_extracted.csv')

case_number = []
dur = []
for index, row in df.iterrows():
    case_number.append(row['audio'].split('/')[1][:3])
    dur.append(librosa.get_duration(librosa.load('BabyCry4-1/' + row['audio'])[0]))

df['id'] = case_number
df['dur'] = dur

print(len(df.index))

df = df.sort_values(by = ['dur'], ascending=False).drop(df.tail(math.floor(0.1*len(df.index))).index)
print(len(df.index))


test_ids = ['G13', 'J24', 'J31', 'G17', 'G29', 'J23', 'J13', 'G24']
test_df = df.loc[df['id'].isin(test_ids)]
test_df['augmented'] = False
test_df.to_csv('BabyCry4-1/test.csv', index=False)
# print(test_df)

exclude = test_ids 

train_df = df.loc[~df['id'].isin(exclude)]
train_df['augmented'] = False
print(train_df['label'].value_counts())

# train_df.to_csv('BabyCry/train.csv')
# print(train_df['label'].value_counts())

for index, row in train_df.iterrows():
    if row['label'] == 1:
        augments = np.random.randint(6, size=3)
        cnt = 0
        for a in augments:
            audio, sr = librosa.load('BabyCry4-1/' + row['audio'])
            audio = augment_data(a, audio)
            audio_file = row['audio'][:-4] + '_aug' + str(cnt) +'.wav'
            cnt = cnt + 1
            sf.write('BabyCry4-1/' +  audio_file, np.ravel(audio), sr)
            new_row = {'audio': audio_file, 'label': row['label'], 'id': row['id'], 'augmented': True}
            train_df = train_df.append(new_row, ignore_index=True)
    
    elif row['label'] == 0:
        augments = np.random.randint(6, size=1)
        for a in augments:
            audio, sr = librosa.load('BabyCry4-1/' + row['audio'])
            audio = augment_data(a, audio)
            audio_file = row['audio'][:-4] + '_aug.wav'
            sf.write('BabyCry4-1/' + audio_file, np.ravel(audio), sr)
            new_row = {'audio': audio_file, 'label': row['label'], 'id': row['id'], 'augmented': True}
            train_df = train_df.append(new_row, ignore_index=True)
            

train_df.to_csv('BabyCry4-1/train.csv', index=False)