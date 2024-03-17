import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers 
import os
import pandas as pd
from PIL import Image


def createOrLoadModel(ckptPath, inputDims, outputDims, lr):
    ckpts = len(os.listdir(ckptPath))
    if(ckpts == 0):
        print("Initialized new Model")

        #model construction
        model = keras.sequential([
            layers.Input(shape = ((inputDims,))),
            layers.Dense(3000, activation='relu'),
            layers.Dense(1000, activation='relu'),
            layers.Dense(500, activation='relu'),
            layers.Dense(outputDims, activation='relu'),
            layers.Softmax()
        ])

        optim = keras.optimizers.Adam(learning_rate=lr)
        loss_fn = keras.losses.CategoricalCrossentropy()

        model.compile(
            optimizer = optim,
            loss = loss_fn,
            metrics = ['accuracy']
        )

    else:
        print("Loaded Model from "+ckptPath)
        model = keras.models.load_model(ckptPath)
    
    return model

def trainModel(m, ckptPath, X_train, Y_train, X_test, Y_test, epochs, batchSize):
    #checkpointing
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath = ckptPath,
    verbose=1,
    monitor='accuracy',
    mode='max',
    save_best_only=True
    )

    #model training
    history = m.fit(
        X_train, Y_train, 
        epochs = epochs,
        batch_size = batchSize, 
        validation_split = 0,
        verbose =1,
        validation_data = (X_test, Y_test),
        callbacks=[model_checkpoint_callback],
        shuffle = True
    )
    return history

def buildArrays(trainDirPath, testDirPath):
    train_samples = []
    for dir in os.listdir(trainDirPath):
        for imgName in os.listdir(trainDirPath+"/"+dir):
            img = Image.open(trainDirPath+"/"+dir+"/"+imgName)
            np_img = np.array(img)
            np_img = np_img.flatten().reshape(np_img.size,1)
            train_samples.append(np_img)
    train = np.hstack(train_samples)
    print(train.shape)

    test_samples = []
    for imgName in os.listdir(testDirPath):
        img = Image.open(testDirPath+'/'+imgName)
        np_img = np.array(img)
        np_img = np_img.flatten().reshape(np_img.size,1)
        test_samples.append(np_img)
    test = np.hstack(test_samples)
    #print(test.shape)
    return train, test

def checkDataFiles():
    dataPath = "./curTest"
    imgSizes = []
    imgsAbove = []
    for img in os.listdir(dataPath):
        image = Image.open(dataPath+"/"+img)
        np_img = np.array(image)
        #np_img = np_img.flatten().reshape(np_img.size,1)
        #print(type(np_img.shape[0]))
        if(np_img.shape[0] >= 45):
            imgsAbove.append(np_img.shape)
        imgSizes.append(np_img.shape)
        #print(np_img.shape)

    print(len(imgSizes))
    print(len(imgsAbove))
    print(max(set(imgSizes), key=imgSizes.count))


trainPath = "./curTrain"
testPath = "./curTest"
X_train, X_test = buildArrays(trainPath, testPath)
#checkDataFiles()
