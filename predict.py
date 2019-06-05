#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import tensorflow as tf
import keras
import numpy as np
import os
import sys
import cv2

from tensorflow.python.keras.callbacks import TensorBoard
from time import time

# Keras Module
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator


os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'

MODEL_NAME = "model_softmax.h5"
MODEL_PATH = "YOUR_MODEL_PATH"
IMAGE_PATH = 'YOUR_IMAGE_PATH'
IMAGE_SIZE = 64
IMAGE_INPUT = sys.argv[1]

np.random.seed(3)
batch_size = 10

class judge:
    
    def __init__(self):
        self.img = None
        pass

    def loadImage(self, file):
        imageFile = file
        print(FILENAME)
        self.img = cv2.imread(imageFile)
        self.faceDetect()
        cv2.imwrite(IMAGE_PATH+"/1.jpg", self.img)
        
    def faceDetect(self):
        faceCascPath = "HAARCASCADE_PATH"
        faceCascade = cv2.CascadeClassifier(faceCascPath)

        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        try:
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor = 1.1,
                minNeighbors = 5,
                minSize = (50,50),
                flags = cv2.CASCADE_SCALE_IMAGE
            )
            for(x,y,w,h) in faces:
                frame = gray[y:y+h, x:x+w]
                self.img = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
            
            return 0
        except:
            print("face not detect")
            sys.exit(1)
            return 1



if __name__ == "__main__":
    frame = judge()
    frame.loadImage(IMAGE_INPUT)
    datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
            'YOUR PREPROCESS IMAGE PATH',
            target_size=(IMAGE_SIZE, IMAGE_SIZE),    
            batch_size=10,
            color_mode = 'grayscale',
            class_mode='categorical')

    model = keras.models.load_model(MODEL_PATH + MODEL_NAME)
    model.compile(loss='categorical_crossentropy', optimizer ='rmsprop', metrics =['accuracy'])

    print("-- Predict --")
    output = model.predict_generator(test_generator, steps=1)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.8f}".format(x)})
    print(test_generator.class_indices)
    print(output[0])