import os
import librosa 

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio

from tensorflow import keras

# def get_one_hot(targets, nb_classes):
#     res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
#     return res.reshape(list(targets.shape)+[nb_classes])

class BabyCry(keras.utils.Sequence):

    def __init__(self, batch_size, root_dir, csv, shuffle, spec = False, input_shape = (1,1,1)):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.spec = spec
        self.input_shape = input_shape

        self.df = pd.read_csv(os.path.join(root_dir, csv))

        self.df_len = len(self.df.index)
        self.indexes = np.arange(self.df_len)

    def __len__(self):
        return len(self.df.index) // self.batch_size
    
    def __getitem__(self, idx):
        """Returns a tupel (input, target)"""
        batch = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]

        audio_batch = [librosa.load(os.path.join(self.root_dir, path), sr=16000)[0] for path in batch['audio']]

        if self.spec:
            audio_batch = [librosa.feature.melspectrogram(x, sr=16000, n_mels=self.input_shape[1], fmax=8000)[:self.input_shape[0]] for x in audio_batch]
        else:
            max_len = max(len(row) for row in audio_batch)
            audio_batch = np.array([np.pad(row, (0, max_len-len(row))) for row in audio_batch])

        label_batch = np.asarray([label for label in batch['label']])


        return audio_batch, label_batch


    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(self.df_len)
        if self.shuffle:
            np.random.shuffle(self.indexes)