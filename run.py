import numpy as np
import tensorflow as tf

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

    [print(i.shape, i.dtype) for i in model.inputs]
    [print(o.shape, o.dtype) for o in model.outputs]
    [print(l.name, l.input_shape, l.dtype) for l in model.layers]

    if config.mode == 'train':
        model.fit(train_ds, batch_size=config.data.batch_size, epochs=config.train_config.epochs)
        


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default_experiment.yaml')

    args = parser.parse_args()
    main(args)