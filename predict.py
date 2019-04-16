# -*- coding: utf-8 -*-
""" This program uses the trained model to predict images into different classes
The program requires the data tree to be of the following format.
|dataset/
    |train/
        |class-A/
        |class-B/
    
    |test/
        |class-A/
        |class-B/
        
To train a simple classification network use the following command
python train.py
:args (optional) -l 3 # The layer that needs to be visualized
"""



import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential, load_model, Model
import numpy as np
from random import shuffle
from random import choice

import matplotlib.pyplot as plt
from tqdm import tqdm
import os, glob, cv2, argparse
import keract

def predict(model_path = 'model.h5', data_path = 'dataset/test_set/', image_size = (64,64,3), n_images = 5, layer_visu = None):
    """
    _func_: Predicts the object in the image using the trained model

    Input:
      model: traned model file
      path(optional): path to the testing dataset folder
      n_images(optional): number of images to be tested
      input_size(optional) - input size of teh image, defaulted to 64x64x3
      layer_visu(optional) - number of convolutional block, Each block has a conv2D->conv2d->Maxpool->batchNorm
    
    Return:
      model: Model file that represents the weights
    """

    # get a list of all the images in the folder
    image_paths = glob.glob(data_path + '*')
    shuffle(image_paths)

    # Load the saved CNN
    model = load_model(model_path)

        # Iterate through every image
    for i in range(n_images):
        
        image_path = choice(image_paths)
        image = cv2.imread(image_path)
        image_predict = cv2.resize(image,(image_size[0],image_size[1]))
        # Resize the image to the one that was used to train the network
        image_predict = np.expand_dims(image_predict, axis=0)
        # Predict the passed image using the trained model
        model_prediction = model.predict(image_predict)

        if model_prediction[0][0] == 1: predicted_label = 1
        else:predicted_label = 0
        
        print(predicted_label)
        cv2.imshow('output', image)
        cv2.waitKey(0)

        visualize_activations(model, image_predict, 5)


        # activations = activation_model.predict(image_predict) 
        # # Returns a list of five Numpy arrays: one array per layer activation

        # first_layer_activation = activations[10]
        # print(first_layer_activation.shape)

        # plt.matshow(first_layer_activation[0, :, :, 0], cmap='viridis')
        # plt.show()
    
def visualize_activations(model, image, layer_number):

    # These are the names of the layers, so can have them as part of our plot
    layer_names = []

    for layer in model.layers[:8]:
        layer_names.append(layer.name)

    images_per_row = 16

    layer_outputs = [layer.output for layer in model.layers]

    activation_model = Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input

    activations = activation_model.predict(image)

    # Now let's display our feature maps
    for layer_name, layer_activation in zip(layer_names, activations):
        # This is the number of features in the feature map
        n_features = layer_activation.shape[-1]

        # The feature map has shape (1, size, size, n_features)
        size = layer_activation.shape[1]

        # We will tile the activation channels in this matrix
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        # We'll tile each filter into this big horizontal grid
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                                :, :,
                                                col * images_per_row + row]
                # Post-process the feature to make it visually palatable
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size,
                            row * size : (row + 1) * size] = channel_image

        # Display the grid
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        
    plt.show()


if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description='Run MURA experiment')
    # parser.add_argument('--batch_size', default=8, type=int, #default=64
    #                     help='Batch size')
    # parser.add_argument('--plot_architecture', type=bool, default=False,#default=False,
    #                     help='Save a plot of the network architecture')

    image_size = (64,64,3)

    predict(n_images = 10)
    