import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_addons as tfa

from keras import backend as K
from keras import layers, losses

def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def trill(model = 'https://tfhub.dev/google/trillsson5/1', bilstm = False, dropout = 0.2):
  input = layers.Input(shape=(None,))
  m = hub.KerasLayer(model)

  # NOTE: Audio should be floats in [-1, 1], sampled at 16kHz. Model input is of
  # the shape [batch size, time].
  # audio_samples = tf.zeros([3, 64000])

  embeddings = m(input)['embedding']
  embeddings = tf.expand_dims(embeddings, axis=1)
  x = layers.Conv1D(1, 1)(embeddings)
  x = layers.Dropout(dropout)(x) # embeddings

  if bilstm:
    x = tf.expand_dims(x, axis=1)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Flatten()(x)
  else:
    x = layers.Flatten()(x) 
    x = layers.Dense(512, activation='relu')(x)

  x = layers.Dense(512, activation='relu')(x)
  predictions = layers.Dense(1, activation='sigmoid')(x)

  trill_pretrained = tf.keras.Model(inputs = m.input, outputs = predictions)
  trill_pretrained.compile(optimizer='adam', loss=losses.BinaryCrossentropy(), metrics=[get_f1, 'accuracy'])
  return trill_pretrained


def resnet(input_shape):
  base_model = tf.keras.applications.ResNet152(weights='imagenet', include_top=False, input_shape=input_shape)
  for layer in base_model.layers:
    layer.trainable = False
  
  x = layers.Flatten()(base_model.output)
  x = layers.Dense(1000, activation='relu')(x)
  predictions = layers.Dense(1, activation='sigmoid')(x)

  resnet_pretrained = tf.keras.Model(inputs=base_model.input, outputs=predictions)
  resnet_pretrained.compile(optimizer='adam', loss=losses.BinaryCrossentropy(), metrics=[get_f1])
  return resnet_pretrained
