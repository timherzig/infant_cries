import os
import librosa 

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio

from tensorflow import keras

class BabyCry(keras.utils.Sequence):

    def __init__(self, batch_size, root_dir, shuffle):
        self.shuffle = shuffle
        self.batch_size = batch_size

        self.df = pd.read_csv(os.path.join(root_dir, 'babycry_extracted.csv'))

        self.df_len = len(self.df.index)
        self.indexes = np.arange(self.df_len)

    def __len__(self):
        return len(self.df.index) // self.batch_size
    
    def __getitem__(self, idx):
        """Returns a tupel (input, target)"""
        batch = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]

        audio_batch = [librosa.load(path, sr=16000)[0] for path in batch['audio']]

        max_len = max(len(row) for row in audio_batch)
        audio_batch = np.array([np.pad(row, (0, max_len-len(row))) for row in audio_batch])

        label_batch = np.asarray([label for label in batch['label']])

        # TODO: figure out the dimensionalities
        print('-----------------------------------------')
        print(audio_batch.shape)
        print(label_batch.shape)
        print('-----------------------------------------')

        return audio_batch, label_batch


    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(self.df_len)
        if self.shuffle:
            np.random.shuffle(self.indexes)