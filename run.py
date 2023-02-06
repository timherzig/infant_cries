import numpy as np
import tensorflow as tf

from omegaconf import OmegaConf
from argparse import ArgumentParser

#Local
from model.trill import trill


def main(args):
    config = OmegaConf.load(args.config)
    
    model = trill(config.model.name)

    





if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default_experiment.yaml')

    args = parser.parse_args()
    main(args)