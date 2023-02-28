import os
import numpy as np
import tensorflow as tf
import keras_tuner as kt
import tensorflow_hub as hub

from omegaconf import OmegaConf
from argparse import ArgumentParser

#Local
from model.models import trill, resnet
from data.babycry import BabyCry

def main(args):

    ## For some reason I need to add this block to make it work with my GPU
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.InteractiveSession(config=config)

    config = OmegaConf.load(args.config)
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    checkpoint_dir = os.path.join('checkpoints', args.config.split('/')[-1][:-5]) # f'version_{len(os.listdir("checkpoints/")) + 1}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    OmegaConf.save(config=config, f=os.path.join(checkpoint_dir, 'config.yaml'))

    test_ds = []
    full_test = BabyCry(config.data.batch_size,
                    config.data.root_dir,
                    'test.csv',
                    True,
                    True if config.model.name == 'resnet' else False,
                    input_shape=(config.model.h, config.model.w, 3))

    for id in os.listdir(config.data.root_dir + '/test/'):
        ds = BabyCry(config.data.batch_size,
                        config.data.root_dir,
                        os.path.join('test', id),
                        True,
                        True if config.model.name == 'resnet' else False,
                        input_shape=(config.model.h, config.model.w, 3))
        id = id[:-4]

        test_ds.append((id, ds))


    f1s = 0
    acc = 0

    if config.model.name == 'resnet':
        model = resnet(input_shape = (config.model.h, config.model.w, 3))
    else:
        model = trill(config.model.name, config.model.bilstm, config.model.dropout)

    train_ds = BabyCry(config.data.batch_size, 
                        config.data.root_dir, 
                        'train-single.csv',
                        True,
                        True if config.model.name == 'resnet' else False,
                        input_shape=(config.model.h, config.model.w, 3))
    
    val_ds = BabyCry(config.data.batch_size, 
                        config.data.root_dir, 
                        'val-single.csv',
                        True,
                        True if config.model.name == 'resnet' else False,
                        input_shape=(config.model.h, config.model.w, 3))
    
    tuner = kt.Hyperband(model,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3)
    
    tuner.search(train_ds, epochs=50, validation_data=val_ds, callbacks=[callback])

    # Get the optimal hyperparameters
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    
    history = model.fit(train_ds,
                validation_data=val_ds,
                batch_size=config.data.batch_size,
                callbacks=[callback],
                epochs=config.train_config.epochs)
    
    
    f = open(checkpoint_dir + '/result.txt', "a")
    f.write(str(history.history))
    f.write('\n\n')
    f.close()

    for t in test_ds:
        test_results = model.evaluate(t[1], batch_size=config.data.batch_size)
        f = open(checkpoint_dir + '/result.txt', "a")
        f.write(f'{str(t[0])} Test loss, test f1, test accuracy: {str(test_results)}\n\n')
        f.close()
    
    test_results = model.evaluate(full_test, batch_size=config.data.batch_size)
    f = open(checkpoint_dir + '/result.txt', "a")
    f.write(f'Full test loss, test f1, test accuracy: {str(test_results)}\n\n')
    f.close()
    f1s = f1s + (test_results[1])
    acc = acc + (test_results[2])

    model.save(save_model_dir)
    
    f = open(checkpoint_dir + '/result.txt', "a")
    f.write(f'Average test results: {str(f1s/config.data.n_fold)}\n')
    f.write(f'Average accuracy results: {str(acc/config.data.n_fold)}\n')
    f.close()
    # ------------------------


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default_experiment.yaml')

    args = parser.parse_args()
    main(args)