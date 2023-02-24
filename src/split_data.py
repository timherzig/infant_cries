import random
import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from augment_data import augment_data

pd.set_option("display.max_rows", None)

df = pd.read_csv('BabyCry/babycry_extracted.csv')

case_number = []
for index, row in df.iterrows():
    case_number.append(row['audio'].split('/')[1][:3])

df['id'] = case_number

# G13 37
# J24 36
# G29 22
# J23 23
# J13 17
# G31 15

test_ids = ['G13', 'J24', 'G29', 'J23', 'J13', 'G31']
test_df = df.loc[df['id'].isin(test_ids)]
test_df['augmented'] = False
test_df.to_csv('BabyCry/test.csv', index=False)
# print(test_df)

train_df = df.loc[~df['id'].isin(test_ids)]
train_df['augmented'] = False

# train_df.to_csv('BabyCry/train.csv')
# print(train_df['label'].value_counts())

for index, row in train_df.iterrows():
    if row['label'] == 1:
        augments = np.random.randint(3, size=2)
        cnt = 0
        for a in augments:
            audio, sr = librosa.load('BabyCry/' + row['audio'])
            audio = augment_data(a, audio)
            audio_file = row['audio'][:-4] + '_aug' + str(cnt) +'.wav'
            cnt = cnt + 1
            sf.write('BabyCry/' +  audio_file, np.ravel(audio), sr)
            new_row = {'audio': audio_file, 'label': row['label'], 'id': row['id'], 'augmented': True}
            train_df = train_df.append(new_row, ignore_index=True)
    
    elif row['label'] == 0:
        augments = np.random.randint(3, size=1)
        for a in augments:
            if (bool(random.getrandbits(1))):
                audio, sr = librosa.load('BabyCry/' + row['audio'])
                audio = augment_data(a, audio)
                audio_file = row['audio'][:-4] + '_aug.wav'
                sf.write('BabyCry/' + audio_file, np.ravel(audio), sr)
                new_row = {'audio': audio_file, 'label': row['label'], 'id': row['id'], 'augmented': True}
                train_df = train_df.append(new_row, ignore_index=True)
            

train_df.to_csv('BabyCry/train.csv', index=False)