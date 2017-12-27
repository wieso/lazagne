import numpy as np
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import adam
from matplotlib import pyplot as plt
from nolearn.lasagne import NeuralNet
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def plot_loss(net):
    train_loss = [row['train_loss'] for row in net.train_history_]
    valid_loss = [row['valid_loss'] for row in net.train_history_]
    plt.plot(train_loss, label='train loss')
    plt.plot(valid_loss, label='test loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    return plt


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


def regr(X, y, X_valid, y_valid):
    l = InputLayer(shape=(None, X.shape[1]))
    l = DenseLayer(l, num_units=100, nonlinearity=softmax)
    # l = DenseLayer(l, num_units=40, nonlinearity=softmax)
    l = DenseLayer(l, num_units=len(np.unique(y)), nonlinearity=softmax)
    net = NeuralNet(l, update=adam, update_learning_rate=0.01, max_epochs=2000, objective_loss_function=squared_error,
                    regression=True)
    net.fit(X, y)
    print(net.score(X, y))
    y_pred = net.predict(X_valid)
    print(y_valid)
    print(y_pred)
    plot_loss(net)
    plt.title('Digits')
    plt.show()


def find_digits(X, y, X_valid, y_valid):
    max_hidden_layers = 1
    max_neuron_units = 1000

    loss = []
    # history = list(range())
    # 0, 0
    l = InputLayer(shape=(None, X.shape[1]))
    l = DenseLayer(l, num_units=len(np.unique(y)), nonlinearity=softmax)
    net = NeuralNet(l, update=adam, update_learning_rate=0.01, max_epochs=2000)
    net.fit(X, y)
    y_pred = net.predict(X_valid)
    loss_error = mean_squared_error(y_valid, y_pred)
    loss_net = (0, 0, loss_error)
    print(loss_net)
    loss.append(loss_net)

    for i in range(1, max_hidden_layers):
        for j in range(1, max_neuron_units, 4):
            l = InputLayer(shape=(None, X.shape[1]))
            for k in range(i):
                l = DenseLayer(l, num_units=j, nonlinearity=softmax)
            l = DenseLayer(l, num_units=len(np.unique(y)), nonlinearity=softmax)
            net = NeuralNet(l, update=adam, update_learning_rate=0.01, max_epochs=2000)

            net.fit(X, y)
            y_pred = net.predict(X_valid)
            loss_error = mean_squared_error(y_valid, y_pred)
            loss_net = (i, j, loss_error)
            print(loss_net)
            loss.append(loss_net)

    print(min(loss, key=lambda x: x[2]))
    # plot_loss(net)
    # plt.title('Digits')
    # plt.show()


def main():
    # DIGITS
    dataset = datasets.load_digits()
    X, X_valid, y, y_valid = train_test_split(dataset.data, dataset.target, test_size=0.2,
                                              random_state=42)
    y = y.astype(np.int32)
    # digits(X, y, X_valid, y_valid)
    find_digits(X, y, X_valid, y_valid)

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
