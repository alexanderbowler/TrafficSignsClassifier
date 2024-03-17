import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers 
import os
#import keras2onnx


print(tf.random.uniform(shape=[2]))
print("\n\n\n\n\n\n\n\n OUTPUT:")

def plotPoints(points1, points2, overlay=False):
    if overlay:
        fig, ax = plt.gcf(), plt.gca()
    else:
        fig, ax = plt.subplots()

    plt.scatter(
        points1[0,:], points1[1,:], 
        color = 'blue', 
        label='Class 1', 
        alpha=0.7
    )
    plt.scatter(
        points2[0,:], points2[1,:],
        color = 'red', 
        label = 'Class 2',
        alpha = 0.7
    )

    ax.legend()
    ax.grid(alpha = 0.2)
    return fig

def randomCircularCoordinates(radius, num_points, sigma = 0.1):
    angles = np.linspace(0, 2* np.pi, num = num_points)

    radii = radius + sigma * np.random.randn(num_points)

    x_coords = radii * np.cos(angles)
    y_coords = radii * np.sin(angles)

    coords = np.vstack((x_coords, y_coords))

    return coords

def coords2output(m, x1, x2):
    x = (np.array([x1,x2]))
    x = np.expand_dims(x, axis=0)
    y = m(x)
    return y.numpy().flatten()

def plotDecisionBoundary(m, x1range, x2range, threshold = 0.0):
    output = np.zeros((x2range.size, x1range.size))
    for i, x1 in enumerate(x1range):
        for j, x2 in enumerate(x2range):
            output[j,i] = coords2output(m, x1, x2)[0]
    decision = np.sign(output-threshold)
    fig, ax = plt.subplots()
    cs = ax.contourf(x1range, x2range, decision, cmap='Greys', alpha=0.5)
    return fig



num_pts = 100
radius1 = 2
circle1 = randomCircularCoordinates(radius1, num_pts)

radius2 = 0.5
circle2 = randomCircularCoordinates(radius2, num_pts)

# print(circle1.shape)

# fig = plotPoints(circle1, circle2)
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.figure(figsize=(100,100))
# fig.savefig('Circles.png')

X_train = np.concatenate((circle1, circle2), axis=1)
X_train = X_train.T

y1, y2 = -1.0, 1.0
Y1_train = np.full((num_pts, 1), y1)
Y2_train = np.full((num_pts, 1), y2)

Y_train = np.concatenate((Y1_train, Y2_train), axis = 0)
threshold = 0.0

checkpoint_filepath = "./tmp/ckpt/checkpoint.model.keras"
ckpts = len(os.listdir("./tmp/ckpt"))
print(ckpts)

if(ckpts == 0):
    print("Initialized new model")

    #model shape construction
    model  = keras.Sequential([
        layers.Input(shape = (2,)),
        layers.Dense(4, activation = 'relu'),
        layers.Dense(1)
    ])

    mse = tf.keras.losses.MeanSquaredError()
    lr = 1e-3
    optim = tf.keras.optimizers.Adam(learning_rate = lr)


            #model params construction
    model.compile(
        optimizer = optim, 
        loss = 'mse',
        metrics = ['mse']
    )

else:
    print("Loaded Model")
    model = keras.models.load_model(checkpoint_filepath)

#checkpointing
epochs = 100

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_filepath,
    verbose=1,
    monitor='mse',
    mode='min',
    save_best_only=True
)


#model training


batch_size = 128
history = model.fit(
    X_train, Y_train, 
    epochs = epochs,
    batch_size = batch_size, 
    validation_split = 0,
    verbose =1,
    callbacks=[model_checkpoint_callback]
)

#concentricModel = keras2onnx.convert_keras(model)
#keras2onnx.save_model(concentricModel, 'my_concentric_model.onnx')

x1range = np.linspace(-3, 3, 200)
x2range = np.linspace(-3, 3, 200)


# fig = plotDecisionBoundary(model, x1range, x2range, threshold)
# plotPoints(circle1, circle2, overlay=True)
# fig.savefig('TrainedModel')
