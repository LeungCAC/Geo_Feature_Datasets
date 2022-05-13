import numpy as np
import tensorflow as tf
import bone_dataset_remaug as dra
import dataset_visual as dv
import data_load as dl
import network as fnn
from multiprocessing.pool import Pool
import os
from PIL import Image
import imageio
from sklearn.model_selection import train_test_split
from multiprocessing import Manager



dv.make_fashion_mnist(True)


def load_fashion_mnist_random(path = '', percentage = 1):
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    if os.path.exists(path + "image_list.txt"):
        os.remove(path + "image_list.txt")
    labellist = os.listdir(path)
    labellist = [int(x) for x in labellist]
    #print(labellist)
    X = []
    Y = []
    for label in labellist:
        X1, Y1 = dl.img_load(path, label)
        x_train, X_test, y_train, Y_test = train_test_split(X1, Y1, test_size=1-percentage, random_state=42)
        X.append(x_train)
        Y.append(y_train)

    x_train = X[0]
    y_train = Y[0]
    for i in range(len(X)-1):
        x_train = np.concatenate((x_train, X[i + 1]), axis = 0)
        y_train = np.concatenate((y_train, Y[i + 1]), axis = 0)
    x_train, X_test, y_train, Y_test = train_test_split(x_train, y_train, test_size = 1 - percentage, random_state = 42)
    x_train = np.concatenate((x_train, X_test), axis = 0)
    y_train = np.concatenate((y_train, Y_test), axis = 0)

    return (x_train, y_train), (x_test, y_test)


def load_mnist_random(path = '', percentage = 1):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if os.path.exists(path + "image_list.txt"):
        os.remove(path + "image_list.txt")
    labellist = os.listdir(path)
    labellist = [int(x) for x in labellist]
    #print(labellist)
    X = []
    Y = []
    for label in labellist:
        X1, Y1 = dl.img_load(path, label)
        x_train, X_test, y_train, Y_test = train_test_split(X1, Y1, test_size=1-percentage, random_state=42)
        X.append(x_train)
        Y.append(y_train)

    x_train = X[0]
    y_train = Y[0]
    for i in range(len(X)-1):
        x_train = np.concatenate((x_train, X[i + 1]), axis = 0)
        y_train = np.concatenate((y_train, Y[i + 1]), axis = 0)
    x_train, X_test, y_train, Y_test = train_test_split(x_train, y_train, test_size = 1 - percentage, random_state = 42)
    x_train = np.concatenate((x_train, X_test), axis = 0)
    y_train = np.concatenate((y_train, Y_test), axis = 0)

    return (x_train, y_train), (x_test, y_test)

def main():

    
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    loss = 0
    acc = 0
    for i in range(10):
        eval_value_n = fnn.mnist_fnn(x_train, y_train, x_test, y_test)
        loss = loss + eval_value_n[0]
        acc = acc + eval_value_n[1]
    print(loss/10, acc/10)

    
    rem_list = [0.6]
    loss_list = []
    acc_list = []
    for per in rem_list:
        (x_train, y_train), (x_test, y_test) = load_mnist_random(path='../datas/mnist/train/', percentage=per)
        loss = 0
        acc = 0
        print(np.shape(x_train)[0])
        for  i in range (10):
            eval_value_n = fnn.mnist_fnn(x_train, y_train, x_test, y_test)
            loss = loss + eval_value_n[0]
            acc = acc + eval_value_n[1]
        loss_list.append(loss/10)
        acc_list.append(acc/10)
    print(loss_list)
    print(acc_list)
    



if __name__ == "__main__":
    main()