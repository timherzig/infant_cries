#
from argparse import ArgumentParser
import plotly.io as pio
pio.renderers.default = "png"

from PIL import Image as PilImage

import numpy as np
import tensorflow as tf
from tensorflow import keras


import tensorflow as tf
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.models import Model


#=================================================================
# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def main(args):
    model_builder = keras.applications.xception.Xception
    img_size = (299, 299)
    preprocess_input = keras.applications.xception.preprocess_input
    decode_predictions = keras.applications.xception.decode_predictions

    #last_conv_layer_name = "block14_sepconv2_act"
    last_conv_layer_name = "dense"  #Existing layers are: ['input_1', 'keras_layer', 'dropout', 'flatten', 'dense', 'dense_1', 'dense_2']


    # The local path to our target image
    #img_path = keras.utils.get_file(
    #    "african_elephant.jpg", "https://i.imgur.com/Bvro0YD.png"
    #)

    #img_path2 = './eagle.jpg'
    #display(Image(img_path2))

    def get_img_array(img_path2, size):
        # `img` is a PIL image of size 299x299
        img = keras.preprocessing.image.load_img(img_path2, target_size=size)
        #img2 = Image(PilImage.open('./eagle.jpg').convert('RGB'))
        #img = keras.preprocessing.image.load_img(img2, target_size=size)

        # `array` is a float32 Numpy array of shape (299, 299, 3)
        array = keras.preprocessing.image.img_to_array(img)
        # We add a dimension to transform our array into a "batch"
        # of size (1, 299, 299, 3)
        array = np.expand_dims(array, axis=0)
        return array

    #def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    def make_gradcam_heatmap(specgram, model, last_conv_layer_name, pred_index=None):
        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            #last_conv_layer_output, preds = grad_model(img_array)
            last_conv_layer_output, preds = grad_model(specgram)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    # Prepare image
    #img_array = preprocess_input(get_img_array(img_path2, size=img_size))

    # Make model
    #model = model_builder(weights="imagenet")


    def f1(y_true, y_pred): #TODO: Why no real F1?
        return 1
    
    audio_file = args.audio # "G0200507_aug0.wav"
    y, sr = librosa.load(audio_file, sr=22050)
    S = librosa.feature.melspectrogram(y, sr=sr, n_fft=2048, hop_length=1024, n_mels=224)
    specgram = librosa.power_to_db(S, ref=np.max)
    specgram = specgram[:, :, np.newaxis]

    specgram = image.img_to_array(specgram)
    specgram = preprocess_input(specgram)
    imgplot = plt.imshow(specgram)
    plt.show()

    path1 = args.model # "/home/razieh/ECAPA-TDNN/FOSO_non-s_trillsson5_d2_babycry4/0/"
    #path1 = "/home/razieh/ECAPA-TDNN/FOSO_non-s_trillsson5_d2_bilstm_babycry4/0/"

    model = tf.keras.models.load_model(path1, custom_objects={'f1':f1}, compile=False)
    # Remove last layer's softmax
    model.layers[-1].activation = None

    # Print what the top predicted class is
    # = model.predict(img_array)
    preds = model.predict(specgram)
    #print("Predicted:", decode_predictions(preds, top=1)[0])

    # Generate class activation heatmap
    #heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    heatmap = make_gradcam_heatmap(specgram, model, last_conv_layer_name)

    print(" heatmap succesfully created")

    # Display heatmap
    plt.matshow(heatmap)
    plt.show()

    def save_and_display_gradcam(specgram, heatmap, cam_path="cam.jpg", alpha=0.4):
        # Load the original image
        img = keras.preprocessing.image.load_img(specgram)
        img = keras.preprocessing.image.img_to_array(img)

        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

        # Save the superimposed image
        superimposed_img.save(cam_path)

        # Display Grad CAM
        display(Image(cam_path))


    save_and_display_gradcam(specgram, heatmap)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--audio', type=str)

    args = parser.parse_args()
    main(args)

"""
#================================

# Load pre-trained ResNet model

base_model = ResNet50(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('conv5_block3_out').output)

# Load audio file and convert to spectrogram
audio_file = "00001.wav"
y, sr = librosa.load(audio_file, sr=22050)
S = librosa.feature.melspectrogram(y, sr=sr, n_fft=2048, hop_length=1024, n_mels=224)
specgram = librosa.power_to_db(S, ref=np.max)
specgram = specgram[:, :, np.newaxis]
specgram = image.img_to_array(specgram)
specgram = preprocess_input(specgram)

# Forward pass through ResNet
preds = model.predict(specgram[np.newaxis, ...])

# Get predicted class and compute gradients
class_idx = np.argmax(preds[0])
grad_model = Model([model.inputs], [model.get_layer('conv5_block3_out').output, model.output])
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(specgram[np.newaxis, ...])
    loss = predictions[:, class_idx]
grads = tape.gradient(loss, conv_outputs)[0]

# Compute weights as average of gradients
weights = np.mean(grads, axis=(0, 1))

# Get feature maps from last convolutional layer
cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)
cam = np.maximum(cam, 0)

# Normalize Grad-CAM
cam = cv2.resize(cam, (224, 224))
cam = cam / np.max(cam)

# Load original audio file
audio, sr = librosa.load(audio_file)

# Plot original audio spectrogram and Grad-CAM overlay
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
librosa.display.specshow(specgram.squeeze(), y_axis='mel', sr=sr, hop_length=1024)
axs[0].set_title("Original Spectrogram")
axs[1].imshow(cv2.cvtColor(cv2.imread("path/to/imagenet_labels.png"), cv2.COLOR_BGR2RGB))
axs[1].imshow(cv2.resize(cam, (224, 224)), alpha=0.5, cmap='jet', interpolation='nearest')
axs[1].set_title("Grad-CAM Overlay")
plt.show()

"""
