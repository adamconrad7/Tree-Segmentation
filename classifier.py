import numpy as np
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from os import listdir
from tifffile import imread, imwrite
import random
# from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay



def augment(ims, labs):

    x = []
    y = []
    for i in range(0, len(ims)):
        r90 = np.rot90(ims[i])
        r180 = np.rot90(r90)
        r270 = np.rot90(r180)
        x.append(ims[i])
        x.append(r90)
        x.append(r180)
        x.append(r270)
        x.append(np.flipud(ims[i]))
        x.append(np.fliplr(ims[i]))
        x.append(np.flipud(r90))
        x.append(np.flipud(r180))

        for j in range(0, 8):
            y.append(labs[i])


    return [np.asarray(x), np.asarray(y)]

def read_segs(path):
    ims = []
    labels = []
    for classname in listdir(path):
        classpath = os.path.join(path, classname)
        for file in listdir(classpath):
            seg = imread(os.path.join(classpath, file))
            ims.append(seg)
            labels.append(classname)

    tup = [np.array(ims), np.array(labels)]
    return tup

def main():
## Based on this tutorial:
##    https://medium.com/@sandy_lee/how-to-train-neural-networks-for-image-classification-part-1-21327fe1cc1

## path to labeled sets
    path = 'training/'

## 80% of data for training
    train_test_split = .8

## 10% of training data for validation
    validation_split = .1

    data = read_segs(path)
    ims = data[0]
    labels = data[1]

## Remove excess objects, might fail if file structure differs
    #print(ims)
    #print(labels[199:])
    ims = np.delete(ims, np.s_[225:], 0)
    ims = np.delete(ims, np.s_[119:175], 0)
    ims = np.delete(ims, np.s_[0:19], 0)
    labels = np.delete(labels, np.s_[225:])
    labels = np.delete(labels, np.s_[119:175])
    labels = np.delete(labels, np.s_[0:19])
    #print(ims)
    #print(labels)

##  Makes string labels into ints
    lookupTable0, idx,  labels, counts = np.unique(labels, return_inverse=True, return_counts=True, return_index=True)

## Adds data
    x, y = augment(ims, labels)

## Shuffles data
    idx = np.random.permutation(len(x))
    X,Y = x[idx], y[idx]

    x_train, x_test= X[:int(X.shape[0]*train_test_split)], X[int(X.shape[0]*train_test_split):]
    y_train, y_test = Y[:int(Y.shape[0]*train_test_split)], Y[int(Y.shape[0]*train_test_split):]

## Scales data from 0-255 to 0-1, surprisingly important for some reason
    x_train, x_test = x_train/ 255., x_test/ 255.

## Neural net, copied structure from tutorial
    model = keras.models.Sequential([keras.layers.Flatten(input_shape = [2,2,4]),
        keras.layers.Dense(300, activation = 'relu' ),
        keras.layers.Dense(100, activation = 'relu' ),
        keras.layers.Dense(100, activation = 'relu' ),
        keras.layers.Dense(100, activation = 'relu' ),
        keras.layers.Dense(6, activation = 'softmax' )])

    model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
        # optimizer = 'RMSprop',  #.9-.97
        # optimizer = 'Adam',     #.95-.97
        # optimizer = 'Adamax',   #.95-.97
        optimizer = 'Nadam',    #.97-.98
        metrics = ['accuracy']
    )

    history = model.fit(x_train, y_train, epochs = 100, validation_split=validation_split)

    print("Evaluating: \n\n\n")
    metrics = model.evaluate(x_test, y_test)
    print(metrics[1])

    y_pred = model.predict_classes(x_test)
    # print(y_pred.shape)
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred.round(), normalize='true')
    print(lookupTable0)
    # confusion_matrix = sklearn.metrics.confusion_matrix(y_test, np.rint(y_pred))
    # print(cm)
    # plot_confusion_matrix(model, x_test, y_test)  # doctest: +SKIP
    # plt.show()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=lookupTable0,

                                      )


    # NOTE: Fill all variables here with default values of the plot_confusion_matrix
    disp = disp.plot(cmap='gray')

    plt.show()

    if metrics[1] > .985:
        model.save('model/')



main()
