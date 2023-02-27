import random
import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from augment_data import augment_data

pd.set_option("display.max_rows", None)

df = pd.read_csv('BabyCry5/babycry_extracted.csv')

case_number = []
for index, row in df.iterrows():
    case_number.append(row['audio'].split('/')[1][:3])

df['id'] = case_number

# G13 37
# J24 36
# J31 31
# G21 30
# G29 22
# J23 23
# G24 17
# J13 17

# EXCLUDE
# G31    15
# G07    15
# G28    14
# G19    14
# G30    13
# G03    13
# G09    13
# G16    13
# G08    13
# G22    12
# J28    11
# G26    11
# G23    11
# G05    10
# G15    10
# G33    10
# G12    10
# G27     9
# G14     9
# G11     9
# J09     8
# G18     7
# G04     6

test_ids = ['G13', 'J03', 'J31', 'G17', 'G29', 'J23', 'J13', 'G24']
test_df = df.loc[df['id'].isin(test_ids)]
test_df['augmented'] = False
test_df.to_csv('BabyCry5/test.csv', index=False)
# print(test_df)

exclude = test_ids # + ['G31','G07','G28','G19','G30','G03','G09','G16','G08','G22','J28','G26','G23','G05','G15','G33','G12','G27','G14','G11','J09','G18','G04']

train_df = df.loc[~df['id'].isin(exclude)]
train_df['augmented'] = False
print(train_df['label'].value_counts())

# train_df.to_csv('BabyCry/train.csv')
# print(train_df['label'].value_counts())

for index, row in train_df.iterrows():
    if row['label'] == 1:
        augments = np.random.randint(6, size=5)
        cnt = 0
        for a in augments:
            audio, sr = librosa.load('BabyCry5/' + row['audio'])
            audio = augment_data(a, audio)
            audio_file = row['audio'][:-4] + '_aug' + str(cnt) +'.wav'
            cnt = cnt + 1
            sf.write('BabyCry5/' +  audio_file, np.ravel(audio), sr)
            new_row = {'audio': audio_file, 'label': row['label'], 'id': row['id'], 'augmented': True}
            train_df = train_df.append(new_row, ignore_index=True)
    
    elif row['label'] == 0:
        augments = np.random.randint(6, size=2)
        for a in augments:
            audio, sr = librosa.load('BabyCry5/' + row['audio'])
            audio = augment_data(a, audio)
            audio_file = row['audio'][:-4] + '_aug.wav'
            sf.write('BabyCry5/' + audio_file, np.ravel(audio), sr)
            new_row = {'audio': audio_file, 'label': row['label'], 'id': row['id'], 'augmented': True}
            train_df = train_df.append(new_row, ignore_index=True)
            

train_df.to_csv('BabyCry5/train.csv', index=False)