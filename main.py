from lasagne import layers
from lasagne.updates import nesterov_momentum
from matplotlib import pyplot as plt
from nolearn.lasagne import NeuralNet
from nolearn.lasagne.visualize import plot_loss
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

net = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
    ],
    # layer parameters:
    input_shape=(None, 64),  # 96x96 input pixels per batch
    hidden_num_units=300,  # number of units in hidden layer
    output_nonlinearity=None,  # output layer uses identity function
    output_num_units=10,  # 30 target values

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,  # flag to indicate we're dealing with regression problem
    max_epochs=1000,  # we want to train this many epochs
    verbose=0,
)

dataset = datasets.load_digits()
X_train, X_valid, y_train, y_valid = train_test_split(dataset.data, dataset.target, test_size=0.2,
                                                      random_state=42)

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
net.fit(X_train, y_train)

y_pred = net.predict(X_valid)
print(y_valid)
print(lb.inverse_transform(y_pred))

plot_loss(net)
plt.show()
