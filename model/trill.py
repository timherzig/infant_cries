import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras import layers, losses


def trill(model = 'https://tfhub.dev/google/trillsson5/1'):
  input = layers.Input(shape=(None,))
  m = hub.KerasLayer(model)

  # NOTE: Audio should be floats in [-1, 1], sampled at 16kHz. Model input is of
  # the shape [batch size, time].
  # audio_samples = tf.zeros([3, 64000])

  embeddings = m(input)['embedding']

  x = layers.Flatten()(embeddings)
  x = layers.Dense(1000, activation='relu')(x)
  predictions = layers.Dense(2, activation='softmax')(x)

  trill_pretrained = tf.keras.Model(inputs = m.input, outputs = predictions)
  trill_pretrained.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
  return trill_pretrained