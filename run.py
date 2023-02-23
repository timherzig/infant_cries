import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from omegaconf import OmegaConf
from argparse import ArgumentParser

#Local
from model.trill import trill
from data.load_data import load_dataloader

def main(args):

    ## For some reason I need to add this block to make it work with my GPU
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.InteractiveSession(config=config)

    config = OmegaConf.load(args.config)
    
    model = trill(config.model.name)

    train_ds = load_dataloader(config)

    if config.mode == 'train':
        model.fit(train_ds, batch_size=config.data.batch_size, epochs=config.train_config.epochs)

        checkpoint_dir = os.path.join('checkpoints', f'version_{len(os.listdir("checkpoints/")) + 1}')
        os.makedirs(checkpoint_dir)

        model.save(os.path.join(checkpoint_dir, 'model'))

        OmegaConf.save(config=config, f=os.path.join(checkpoint_dir, 'config.yaml'))
    elif config.mode == 'evaluate':
        print('evaluating')
        


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default_experiment.yaml')

    args = parser.parse_args()
    main(args)