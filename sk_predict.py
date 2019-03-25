import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from data_helper import data_loader, mae

train_x, train_y, test_x, test_y = data_loader()
train_y = np.array(train_y)
test_y = np.array(test_y)
train_y_in, train_y_out = train_y[:, 0], train_y[:, 1]
test_y_in, test_y_out = test_y[:, 0], test_y[:, 1]


def svr_model():
    svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1, verbose=True)
    svr_rbf.fit(train_x, train_y_in)
    pre_y = svr_rbf.predict(test_x)
    print(mae(np.array(pre_y), test_y_in))


def sgd_model():
    sgd = SGDRegressor(loss='huber', max_iter=1000000, learning_rate='invscaling', eta0=20,
                       verbose=True)
    sgd.fit(train_x, train_y_in)
    pre_y = sgd.predict(test_x)
    print(mae(np.array(pre_y), test_y_in))


def nn_model():
    nn = MLPRegressor(hidden_layer_sizes=(200, 100, 2), activation='relu', solver='adam', verbose=True)
    nn.fit(train_x, train_y_in)
    pre_y = nn.predict(test_x)
    print(mae(np.array(pre_y), test_y_in))


nn_model()
