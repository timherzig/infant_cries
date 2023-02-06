import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio

from tensorflow import keras

def load_develop_data():
    audio = tfio.audio.AudioIOTensor('/home/tim/Documents/dfki/ds/cv-corpus-10.0-2022-07-04-pl/cv-corpus-10.0-2022-07-04/pl/clips/common_voice_pl_20547774.mp3')

    # audio = tfio.audio.resample(audio, audio.rate, 16000)

    return audio

class DevelopData(keras.utils.Sequence):
    """Data loader to test implementation"""

    def __init__(self, batch_size, sample_rate, root_dir = '/home/tim/Documents/dfki/ds/cv-corpus-12.0-2022-12-07-nn-NO/cv-corpus-12.0-2022-12-07/nn-NO/', split = 'dev', n_rows = 500, shuffle = True):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        
        self.audio_dir = os.path.join(root_dir, 'audio')
        self.df = pd.read_csv(os.path.join(root_dir, split + '.tsv'), sep='\t')

        self.audio_paths = self.df['path'].to_list()

        self.dflen = len(self.df.index)
        self.indexes = np.arange(self.dflen)

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return len(self.df.index) // self.batch_size
    
    def __getitem__(self, idx):
        """Returns a tupel (input, target)"""

        batch_indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        # audio_location = os.path.join(self.audio_dir, self.df.iloc[batch_indexes, 1])
        audio_location = os.path.join(self.audio_dir, self.audio_paths[batch_indexes])

        audio = tfio.audio.AudioIOTensor(audio_location)

        label = tf.ones(shape=(self.batch_size))

        return audio, label

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(self.dflen)
        if self.shuffle:
            np.random.shuffle(self.indexes)

