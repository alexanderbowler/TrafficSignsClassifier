import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers 
import numpy as np
import ConcentricCircleClassifier as CC

iters = 100
batch_size = 128

model = CC.returnModel()
history = model.fit(
    CC.X_train, CC.Y_train, 
    epochs = iters,
    batch_size = batch_size, 
    validation_split = 0,
    verbose =1
)