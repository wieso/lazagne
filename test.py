import numpy as np
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.nonlinearities import softmax, sigmoid, rectify
from lasagne.updates import adam
from matplotlib import pyplot as plt
from nolearn.lasagne import NeuralNet
from nolearn.lasagne.visualize import plot_loss
from sklearn import datasets
from sklearn.model_selection import train_test_split


def iris(X, y, X_valid, y_valid):
    l = InputLayer(shape=(None, X.shape[1]))
    l = DenseLayer(l, num_units=8, nonlinearity=softmax)
    l = DenseLayer(l, num_units=5, nonlinearity=softmax)
    l = DenseLayer(l, num_units=3, nonlinearity=softmax)
    l = DenseLayer(l, num_units=len(np.unique(y)), nonlinearity=softmax)
    net = NeuralNet(l, update_learning_rate=0.01, max_epochs=10000, )
    net.fit(X, y)
    print(net.score(X, y))
    y_pred = net.predict(X_valid)
    print(y_valid)
    print(y_pred)
    plot_loss(net)
    plt.title('Iris')
    plt.show()


def digits(X, y, X_valid, y_valid):
    l = InputLayer(shape=(None, X.shape[1]))
    l = DenseLayer(l, num_units=100, nonlinearity=softmax)
    # l = DenseLayer(l, num_units=40, nonlinearity=softmax)
    l = DenseLayer(l, num_units=len(np.unique(y)), nonlinearity=softmax)
    net = NeuralNet(l, update=adam, update_learning_rate=0.01, max_epochs=2000)
    net.fit(X, y)
    print(net.score(X, y))
    y_pred = net.predict(X_valid)
    print(y_valid)
    print(y_pred)
    plot_loss(net)
    plt.title('Digits')
    plt.show()


def cancer(X, y, X_valid, y_valid):
    l = InputLayer(shape=(None, X.shape[1]))
    l = DenseLayer(l, num_units=len(np.unique(y)), nonlinearity=softmax)
    net = NeuralNet(l, update_learning_rate=0.01, max_epochs=1000)
    net.fit(X, y)
    print(net.score(X, y))
    y_pred = net.predict(X_valid)
    print(y_valid)
    print(y_pred)
    plot_loss(net)
    plt.title('Cancer')
    plt.show()


def wine(X, y, X_valid, y_valid):
    l = InputLayer(shape=(None, X.shape[1]))
    l = DenseLayer(l, num_units=16, nonlinearity=softmax)
    l = DenseLayer(l, num_units=len(np.unique(y)), nonlinearity=softmax)
    net = NeuralNet(l, update_learning_rate=0.001, max_epochs=1000)
    net.fit(X, y)
    print(net.score(X, y))
    y_pred = net.predict(X_valid)
    print(y_valid)
    print(y_pred)
    plot_loss(net)
    plt.title('Wine')
    plt.show()


def main():
    # Classification with two classes:

    # IRIS
    dataset = datasets.load_iris()
    X, X_valid, y, y_valid = train_test_split(dataset.data, dataset.target, test_size=0.2,
                                              random_state=42)
    y = y.astype(np.int32)
    iris(X, y, X_valid, y_valid)

    # DIGITS
    dataset = datasets.load_digits()
    X, X_valid, y, y_valid = train_test_split(dataset.data, dataset.target, test_size=0.2,
                                              random_state=42)
    y = y.astype(np.int32)
    digits(X, y, X_valid, y_valid)

    # # CANCER
    # dataset = datasets.load_breast_cancer()
    # X, X_valid, y, y_valid = train_test_split(dataset.data, dataset.target, test_size=0.2,
    #                                           random_state=42)
    # y = y.astype(np.int32)
    # cancer(X, y, X_valid, y_valid)
    #
    # # WINE
    # dataset = datasets.load_wine()
    # X, X_valid, y, y_valid = train_test_split(dataset.data, dataset.target, test_size=0.2,
    #                                           random_state=42)
    # y = y.astype(np.int32)
    # wine(X, y, X_valid, y_valid)


if __name__ == '__main__':
    main()
