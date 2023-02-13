import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio


from argparse import ArgumentParser


def main(args):
    
    ds = pd.read_csv(args.csv)

    ds['audio'] = tf.Tensor
    ds['label'] = 0

    audio_list = []
    label_list = []

    for index, row in ds.iterrows():
        path = os.path.join('/home/tim/Documents/dfki/shared_projects/infant_cries/', row['path'])

        # audio = tfio.audio.AudioIOTensor(path, dtype=tf.int16)
        # audio = tf.cast(tf.squeeze(audio.to_tensor()), tf.float32) / 32768.0 

        audio_list.append(path)

        if row['country'] == 'G':
            label_list.append(1)
        else:
            label_list.append(0)

    ds = pd.DataFrame(list(zip(audio_list, label_list)),
               columns =['audio', 'label'])

    ds.to_csv(os.path.join(args.csv[:-12], 'babycry_extracted.csv'), index=False)
    print('done')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--csv', type=str, default='/home/tim/Documents/dfki/shared_projects/infant_cries/BabyCry/babycry.csv')

    args = parser.parse_args()
    main(args)