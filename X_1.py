import numpy as np
import matplotlib.pyplot as plt

def shuffle(x, y):
    r_indexes = np.arange(len(x))
    np.random.shuffle(r_indexes)
    return x[r_indexes], y[r_indexes]

def relu(z):
    return np.maximum(0,z)    

def softmax(z2):
    return np.exp(z2) / np.sum(np.exp(z2)) 

def forward_prop(w1, w2, b1, b2, x_train):
    z1 = w1.dot(x_train) + b1
    a1 = relu(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(a1)
    return z1, a1, z2, a2

def one_hot(y_train):
    one_hot_y = np.zeros((y_train.size, y_train.max() + 1))
    one_hot_y[np.arange(y_train.size),y_train] = 1
    return one_hot_y.T

def deriv_relu(z):
    return z > 0

def back_prop(z1, a1, z2, a2, w2, x_train, y_train):
    m = y_train.size
    one_hot_y = one_hot(y_train)
    dz2 = a2 - one_hot_y
    dw2 = 1 / m * dz2.dot(a1.T)
    db2 = 1 / m * np.sum(dz2, 2)
    dz1 = w2.T.dot(dz2) * deriv_relu(z1)
    dw1 = 1 / m * dz1.dot(x_train.T)
    db1 = 1 / m * np.sum(dz2, 2)
    return dw1, db1, dw2, db2

def actualiza_param(w1, b1, w2, b2, dw1, db1, dw2, db2, lr):
    w1 = w1 - lr * dw1
    b1 = b1 - lr * db1
    w2 = w2 - lr * dw2
    b1 = b2 - lr * db2
    return w1, b1, w2, b2

def get_predictions(a2):
    return np.argmax(a2,0)

def accuracy(predictions,y_train):
    print(predictions,y_train)
    return np.sum(predictions == y_train) / y_train.size

def gradient_descent(w1, w2, b1, b2, x_train, y_train, epochs, lr):
    for i in range(epochs):
            z1, a1, z2, a2 = forward_prop(w1, w2, b1, b2, x_train)
            dw1, db1, dw2, db2 = back_prop(z1, a1, z2, a2, w2, x_train, y_train)
            w1, b1, w2, b2 = actualiza_param(w1, b1, w2, b2, dw1, db1, dw2, db2, lr)

            if i % 50 == 0:
                print("Iteracion: ",i)
                print("Accuracy: ", accuracy(get_predictions(a2), y_train))
    return w1, b1, w2, b2

def main():
    x = np.load('x.npy') / 255
    y = np.load('y.npy')
    x, y = shuffle(x, y)

    m, n, _ = x.shape
    x = x.reshape(m, n * n)
    y = y.reshape(m, 1)

    train_size = int(m * 0.8)
    test_size = int(m * 0.1)

    x_train, x_test = x[:train_size], x[train_size + test_size:]
    y_train, y_test = y[:train_size], y[train_size + test_size:]

    w1 = np.random.rand(10,x_train.shape[1])
    w2 = np.random.rand(10,x_train.shape[1])
    b1 = np.random.rand((m, 1))
    b2 = np.random.rand((m, 1))
    
    z1, a1, z2, a2 = forward_prop(w1, w2, b1, b2, x_train)
    dw1, db1, dw2, db2 = back_prop(z1, a1, z2, a2, w2, x_train, y_train)
    lr = 0.1

    w1, b1, w2, b2 = actualiza_param(w1, b1, w2, b2, dw1, db1, dw2, db2, lr)

    epochs = 1000

    w1, b1, w2, b2 = gradient_descent(w1, w2, b1, b2, x_train, y_train, epochs, lr)

if __name__ == '__main__':
    main()