import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from omegaconf import OmegaConf
from argparse import ArgumentParser

#Local
from model.trill import trill
from data.babycry import BabyCry

def main(args):

    ## For some reason I need to add this block to make it work with my GPU
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.InteractiveSession(config=config)

    config = OmegaConf.load(args.config)
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    checkpoint_dir = os.path.join('checkpoints', f'version_{len(os.listdir("checkpoints/")) + 1}')
    os.makedirs(checkpoint_dir)
    OmegaConf.save(config=config, f=os.path.join(checkpoint_dir, 'config.yaml'))

    test_ds = BabyCry(config.data.batch_size,
                      config.data.root_dir,
                      os.path.join('test.csv'),
                      True)
    f1s = 0

    for i in range(config.data.n_fold):

        model = trill(config.model.name)

        train_ds = BabyCry(config.data.batch_size, 
                           config.data.root_dir, 
                           os.path.join(str(config.data.n_fold) + '_fold_split', str(i), 'train.csv'),
                           True)
        
        val_ds = BabyCry(config.data.batch_size, 
                           config.data.root_dir, 
                           os.path.join(str(config.data.n_fold) + '_fold_split', str(i), 'val.csv'),
                           True)

        history = model.fit(train_ds,
                  validation_data=val_ds,
                  batch_size=config.data.batch_size,
                  callbacks=[callback],
                  epochs=20)
        
        test_results = model.evaluate(test_ds, batch_size=config.data.batch_size)
        
        f = open(checkpoint_dir + '/result.txt', "a")
        f.write(f'Run {i+1} results: \n')
        f.write(history.history)
        f.write('\n\n')
        f.write(f'Test loss, test f1: {test_results}\n\n')
        f.close()

        f1s = f1s + test_results[1]
    
    f = open(checkpoint_dir + '/result.txt', "a")
    f.write(f'Average test results: {f1s/config.data.n_fold}\n')
    f.close()
    # ------------------------


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default_experiment.yaml')

    args = parser.parse_args()
    main(args)