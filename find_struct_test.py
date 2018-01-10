import numpy as np
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import adam
from matplotlib import pyplot as plt
from nolearn.lasagne import NeuralNet
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
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
    max_hidden_layers = 4
    max_neuron_units = 110

    loss = []
    kf = KFold(n_splits=5)
    for i in range(1, max_hidden_layers):
        for j in range((64 + 10) // 2 // i, max_neuron_units // i, 10 // i):
            print('=' * 40)
            print('%s hidden layers' % i)
            print('%s neurons' % j)
            print('=' * 40)
            l = InputLayer(shape=(None, X.shape[1]))
            for k in range(i):
                l = DenseLayer(l, num_units=j, nonlinearity=softmax)
            l = DenseLayer(l, num_units=len(np.unique(y)), nonlinearity=softmax)
            net = NeuralNet(l, update=adam, update_learning_rate=0.01, max_epochs=500)

            k_loss = []
            y_data = np.array([y]).transpose()
            data = np.concatenate((X, y_data), axis=1)
            for train_index, test_index in kf.split(data):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                net.fit(X_train, y_train)
                y_pred = net.predict(X_test)
                loss_error = mean_squared_error(y_test, y_pred)
                k_loss.append(loss_error)
                print(loss_error)

            loss_net = (i, j, np.array(k_loss).mean())
            print(loss_net)
            loss.append(loss_net)
            print('=' * 40)

    print(min(loss, key=lambda x: x[2]))
    # plot_loss(net)
    # plt.title('Digits')
    ##  plt.show()


def smart_find(X, y, X_valid, y_valid):
    loss = []
    kf = KFold(n_splits=5, shuffle=True)
    conf_set = set()
    step = (64 + 10) / 4
    max_neuron_units = step * 8
    for i in range(1, max_neuron_units, step):
        for j in range(0, max_neuron_units, step):
            for k in range(0, max_neuron_units, step):
                struct_net = (i)
                l = InputLayer(shape=(None, X.shape[1]))
                # ------- HIDDEN -----------
                l = DenseLayer(l, num_units=i, nonlinearity=softmax)
                if j > 0:
                    if i + step < j:
                        continue

                    l = DenseLayer(l, num_units=j, nonlinearity=softmax)
                    struct_net = (i, j)
                    if k > 0:
                        if i + step < k or j + step < k:
                            continue
                        struct_net = (i, j, k)
                        l = DenseLayer(l, num_units=k, nonlinearity=softmax)
                # ------- HIDDEN -----------
                l = DenseLayer(l, num_units=len(np.unique(y)), nonlinearity=softmax)
                net = NeuralNet(l, update=adam, update_learning_rate=0.01, max_epochs=250)
                if struct_net in conf_set:
                    continue

                print('=' * 40)
                print(struct_net)
                print('=' * 40)
                conf_set.add(struct_net)

                k_loss = []
                y_data = np.array([y]).transpose()
                data = np.concatenate((X, y_data), axis=1)
                for train_index, test_index in kf.split(data):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    net.fit(X_train, y_train)
                    y_pred = net.predict(X_test)
                    loss_error = net.score(X_test, y_test)
                    # loss_error = mean_squared_error(y_test, y_pred)
                    k_loss.append(loss_error)
                    print(loss_error)

                loss_net = (i, j, k, np.array(k_loss).mean())
                print(loss_net)
                loss.append(loss_net)
                print('=' * 40)

    # for i in range(1, max_hidden_layers):
    #     for j in range((64 + 10) // 2 // i, max_neuron_units // i, 10 // i):
    #         print('=' * 40)
    #         print('%s hidden layers' % i)
    #         print('%s neurons' % j)
    #         print('=' * 40)
    #         l = InputLayer(shape=(None, X.shape[1]))
    #         for k in range(i):
    #             l = DenseLayer(l, num_units=j, nonlinearity=softmax)
    #         l = DenseLayer(l, num_units=len(np.unique(y)), nonlinearity=softmax)
    #         net = NeuralNet(l, update=adam, update_learning_rate=0.01, max_epochs=500)
    #
    #         k_loss = []
    #         y_data = np.array([y]).transpose()
    #         data = np.concatenate((X, y_data), axis=1)
    #         for train_index, test_index in kf.split(data):
    #             X_train, X_test = X[train_index], X[test_index]
    #             y_train, y_test = y[train_index], y[test_index]
    #
    #             net.fit(X_train, y_train)
    #             y_pred = net.predict(X_test)
    #             loss_error = mean_squared_error(y_test, y_pred)
    #             k_loss.append(loss_error)
    #             print(loss_error)
    #
    #         loss_net = (i, j, np.array(k_loss).mean())
    #         print(loss_net)
    #         loss.append(loss_net)
    #         print('=' * 40)

    print(min(loss, key=lambda x: x[3]))
    print(max(loss, key=lambda x: x[3]))
    print(loss)


def net1(X, y, n1):
    l = InputLayer(shape=(None, X.shape[1]))
    # ------- HIDDEN -----------
    l = DenseLayer(l, num_units=n1, nonlinearity=softmax)
    # ------- HIDDEN -----------
    l = DenseLayer(l, num_units=len(np.unique(y)), nonlinearity=softmax)
    net = NeuralNet(l, update=adam, update_learning_rate=0.01, max_epochs=250)
    return net


def net2(X, y, n1, n2):
    l = InputLayer(shape=(None, X.shape[1]))
    # ------- HIDDEN -----------
    l = DenseLayer(l, num_units=n1, nonlinearity=softmax)
    l = DenseLayer(l, num_units=n2, nonlinearity=softmax)
    # ------- HIDDEN -----------
    l = DenseLayer(l, num_units=len(np.unique(y)), nonlinearity=softmax)
    net = NeuralNet(l, update=adam, update_learning_rate=0.01, max_epochs=250)
    return net


def net3(X, y, n1, n2, n3):
    l = InputLayer(shape=(None, X.shape[1]))
    # ------- HIDDEN -----------
    l = DenseLayer(l, num_units=n1, nonlinearity=softmax)
    l = DenseLayer(l, num_units=n2, nonlinearity=softmax)
    l = DenseLayer(l, num_units=n3, nonlinearity=softmax)
    # ------- HIDDEN -----------
    l = DenseLayer(l, num_units=len(np.unique(y)), nonlinearity=softmax)
    net = NeuralNet(l, update=adam, update_learning_rate=0.01, max_epochs=250)
    return net


def net4(X, y, n1, n2, n3, n4):
    l = InputLayer(shape=(None, X.shape[1]))
    # ------- HIDDEN -----------
    l = DenseLayer(l, num_units=n1, nonlinearity=softmax)
    l = DenseLayer(l, num_units=n2, nonlinearity=softmax)
    l = DenseLayer(l, num_units=n3, nonlinearity=softmax)
    l = DenseLayer(l, num_units=n4, nonlinearity=softmax)
    # ------- HIDDEN -----------
    l = DenseLayer(l, num_units=len(np.unique(y)), nonlinearity=softmax)
    net = NeuralNet(l, update=adam, update_learning_rate=0.01, max_epochs=250)
    return net


def net15(X, y):
    l = InputLayer(shape=(None, X.shape[1]))
    # ------- HIDDEN -----------
    l = DenseLayer(l, num_units=1, nonlinearity=softmax)
    l = DenseLayer(l, num_units=1, nonlinearity=softmax)
    l = DenseLayer(l, num_units=1, nonlinearity=softmax)
    l = DenseLayer(l, num_units=1, nonlinearity=softmax)
    l = DenseLayer(l, num_units=1, nonlinearity=softmax)
    l = DenseLayer(l, num_units=1, nonlinearity=softmax)
    l = DenseLayer(l, num_units=1, nonlinearity=softmax)
    l = DenseLayer(l, num_units=1, nonlinearity=softmax)
    l = DenseLayer(l, num_units=1, nonlinearity=softmax)
    l = DenseLayer(l, num_units=1, nonlinearity=softmax)
    l = DenseLayer(l, num_units=1, nonlinearity=softmax)
    l = DenseLayer(l, num_units=1, nonlinearity=softmax)
    l = DenseLayer(l, num_units=1, nonlinearity=softmax)
    l = DenseLayer(l, num_units=1, nonlinearity=softmax)
    l = DenseLayer(l, num_units=1, nonlinearity=softmax)
    # ------- HIDDEN -----------
    l = DenseLayer(l, num_units=len(np.unique(y)), nonlinearity=softmax)
    net = NeuralNet(l, update=adam, update_learning_rate=0.01, max_epochs=250)
    return net


def fit_transform(net, X, y):
    k_loss = []
    y_data = np.array([y]).transpose()
    kf = KFold(n_splits=5, shuffle=True)
    data = np.concatenate((X, y_data), axis=1)
    for train_index, test_index in kf.split(data):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        net.fit(X_train, y_train)
        y_pred = net.predict(X_test)
        loss_error = net.score(X_test, y_test)
        # loss_error = mean_squared_error(y_test, y_pred)
        k_loss.append(loss_error)
        print(loss_error)

    loss_net = np.array(k_loss).mean()
    print('Mean Loss', loss_net)
    print('=' * 40)
    return loss_net


def neurons_find(X, y):
    loss = []
    k_neurons = 15

    print('=' * 40)
    print('net', k_neurons)
    print('=' * 40)
    n1 = net1(X, y, k_neurons)

    loss_net = fit_transform(n1, X, y)
    loss.append(loss_net)

    for i in range(1, k_neurons):
        print('=' * 40)
        print('net', i, k_neurons - i)
        print('=' * 40)
        n2 = net2(X, y, i, k_neurons - i)

        loss_net = fit_transform(n2, X, y)
        loss.append(loss_net)

    for i in range(1, k_neurons - 1):
        for j in range(1, i):
            if i <= 1 \
                    or j <= 1 \
                    or k_neurons - i - j <= 1:
                continue
            print('=' * 40)
            print('net', i, j, k_neurons - i - j)
            print('=' * 40)
            n3 = net3(X, y, i, j, k_neurons - i - j)

            loss_net = fit_transform(n3, X, y)
            loss.append(loss_net)

    for i in range(1, k_neurons):
        for j in range(1, i):
            for k in range(1, j):
                if i <= 1 \
                        or j <= 1 \
                        or k <= 1 \
                        or k_neurons - i - j - k <= 1:
                    continue
                print('=' * 40)
                print('net', i, j, k, k_neurons - i - j - k)
                print('=' * 40)
                n4 = net4(X, y, i, j, k, k_neurons - i - j - k)

                loss_net = fit_transform(n4, X, y)
                loss.append(loss_net)

    print('Min ', min(loss))
    print('Max ', max(loss))
    print(sorted(loss, reverse=True))


def main():
    # DIGITS
    dataset = datasets.load_digits()
    X, y = dataset.data, dataset.target

    pca = PCA(n_components=19)
    X = pca.fit_transform(X)
    y = y.astype(np.int32)
    # digits(X, y, X_valid, y_valid)
    # smart_find(X, y, X_valid, y_valid)
    neurons_find(X, y)

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
