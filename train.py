# -*- coding: utf-8 -*-
""" This program trains a simple two layer classifier for the task of classifying 
elements of the data set.
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
:args (optional) -p dataset path
"""

import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import os, glob, cv2
from random import shuffle
import matplotlib.pyplot as plt

# def load_data(path = 'dataset/train/', image_size = (64,64,3), class_names = ['cat', 'dog'], function='train', train_percnt = 0.80, display=False):
#     """
#     _func_: Loads the training data and returns the images and labels

#     Input:
#         path(optional): path to the training dataset folder
#         input_size(optional) - input size of the image, defaulted to 64x64x3
#         function(optional): 'train' or 'test'

#     Return:
#         x: numpy array of the images
#         y: numpy array of labels labels
#     """

#     # get a list of all the images in the folder
#     image_paths = glob.glob(path + '*')
#     shuffle(image_paths)

#     # Split the training and validation set
#     if function == 'train': image_paths= image_paths[0:int(len(image_paths)*train_percnt)]
#     else: image_paths= image_paths[int(len(image_paths)*train_percnt): len(image_paths)]

#     # Create a vairable for holding the data 
#     # x = np.zeros((len(image_paths), image_size[0], image_size[1], image_size[2]), dtype=np.float32)
#     # y = np.zeros((len(image_paths), 1), dtype=np.float32)
#     x = []
#     y = []

#     print("\n[INFO]............ Loading trainig images\n") 
#     # Iterate through every image
#     for i, image_path in enumerate(tqdm(image_paths)):
        
#         image = cv2.imread(image_path)
#         image = cv2.resize(image,(image_size[0],image_size[1]), interpolation=cv2.INTER_CUBIC)

#         x.append(image)

#         if class_names[0] in image_path: y.append(0)
#         else: y.append(1)
    
#     print("\n[INFO]............ Total of %r images"%(len(image_paths)))
    
#     if display:
#         for i in range(1,5):
#             plt.subplot(2, 2, i)
#             plt.title("True label :"+str(y[i]))  # set title
#             plt.imshow(x[i])
#             plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)
#         plt.show()

#     return np.array(x, dtype="float") / 255.0, np.array(y)

def model_generator(input_size = (64,64,3), n_conv_block = 2, n_dense_block = 2, Dropout=False):
    """
    _func_: Creates a model from the input specifications and returns the model

    Input:
      input_size(optional) - input size of teh image, defaulted to 64x64x3
      n_conv_block(optional) - number of convolutional block, Each block has a conv2D->conv2d->Maxpool->batchNorm
    
    Return:
      model: Model file that represents the weights
    """
    # Initialising the CNN
    model = Sequential()

    # Step 1 - Initial Convolution
    model.add(Conv2D(8, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

    # Iterate through the numer of convolutional blocks
    for i in range(1, n_conv_block):

        # Step 1 - Convolution
        model.add(Conv2D(16*i, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

        # Step 2 - Pooling
        model.add(Conv2D(16*i*2, (3, 3), activation = 'relu'))

        # Pooling again
        model.add(MaxPooling2D(pool_size = (2, 2)))
        
        # Batch Normalization
        model.add(BatchNormalization())

    # Step 3 - Flattening
    model.add(Flatten())

    # Step 4 - Full connection
    # Iterate through the numer of Dense blocks
    for i in range(1, n_dense_block):
        model.add(Dense(units = 128*i, activation = 'relu'))
        if Dropout: model.add(Dropout(0.20))
    model.add(Dense(units = 1, activation = 'sigmoid'))


    # Compiling the CNN
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model

if __name__ == "__main__":
    
    image_size = (64,64,3)
    n_epochs = 5
    batch_size = 32

    train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)

    valid_datagen = ImageDataGenerator(rescale = 1./255)



    training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size = (64, 64),
                                                    batch_size = 32,
                                                    class_mode = 'binary')

    validation_set = valid_datagen.flow_from_directory('dataset/valid_set',
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'binary')

    # Create a Model
    model = model_generator(image_size, n_conv_block=4, n_dense_block=3)
    # model = model_generator()

    # Open the file
    with open('model_report.txt','w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    # Training the model with callback based on validation loss

    model.fit_generator(training_set,
                        epochs = n_epochs,
                        steps_per_epoch = 8000,
                        validation_data = validation_set,
                        validation_steps = 2000,
                        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',
                                min_delta = 0,
                                patience = 2,
                                verbose=0,
                                mode='auto')])

    model.save('model.h5')



