import tensorflow as tf
import tf.keras as keras

class Net(keras.Model):
    """A class to create neural networks that allows for checkpoiting of the models"""

    def __init__(self, inSize, outSize, n):
        super(Net, self).__init__()
        self.dense1 = keras.layers.Dense(inSize, activation = 'relu')
        self.dense2 = keras.layers.Dense(n, activation = 'relu')
        self.dense3 = keras.layers.Dense(n, activation = 'relu')
        self.dense4 = keras.layers.Dense(outSize, activation = 'softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.dense4(x)