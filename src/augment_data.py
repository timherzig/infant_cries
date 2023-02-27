import librosa
import numpy as np
from random import randrange, uniform

def stretch(x):
    rate = uniform(0.7, 1.3)
    return librosa.effects.time_stretch(x, rate)

def random_pad(x):
    x = np.pad(x, (randrange(0, int(0.3*len(x))), randrange(0, int(0.3*len(x)))), 'constant', constant_values=(0,0))
    return x

def add_wn(x):
    noise = np.random.randn(len(x))
    augmented_data = x + 0.05 * noise
    augmented_data = augmented_data.astype(type(x[0]))
    return augmented_data

def augment_data(aug, audio):
    if aug < 4:
        return stretch(audio)
    elif aug < 6:
        return random_pad(audio)
    else:
        return add_wn(audio)