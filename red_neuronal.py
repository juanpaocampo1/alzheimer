import numpy as np
import matplotlib.pyplot as plt

def shuffle(x, y):
    r_indexes = np.arange(len(x))
    np.random.shuffle(r_indexes)
    return x[r_indexes], y[r_indexes]

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

def forward_prop(w1, w2, b1, b2, x_train):
    z1 = np.dot(w1, x_train.T) + b1
    a1 = relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_relu(z):
    return z > 0

def back_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * deriv_relu(Z1)
    dW1 = 1 / m * dZ1.dot(X)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def actualiza_param(w1, b1, w2, b2, dw1, db1, dw2, db2, lr):
    w1 = w1 - lr * dw1
    b1 = b1 - lr * db1
    w2 = w2 - lr * dw2
    b2 = b2 - lr * db2
    return w1, b1, w2, b2

def get_predictions(a2):
    return np.argmax(a2)

def accuracy(predictions, y_train):
    return np.sum(predictions == y_train) / y_train.size

def gradient_descent(w1, w2, b1, b2, x_train, y_train, epochs, lr):
    for i in range(epochs):
        z1, a1, z2, a2 = forward_prop(w1, w2, b1, b2, x_train)
        dw1, db1, dw2, db2 = back_prop(z1, a1, z2, a2, w1, w2, x_train, y_train)
        w1, b1, w2, b2 = actualiza_param(w1, b1, w2, b2, dw1, db1, dw2, db2, lr)

        if i % 50 == 0:
            print("Iteracion: ", i)
            predictions = get_predictions(a2)
            acc = accuracy(predictions, y_train)
            print("Accuracy: ", acc)

    return w1, b1, w2, b2

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2,x_test,y_test):
    current_image = x_test[:, index, None]
    prediction = make_predictions(x_test[:, index, None], W1, b1, W2, b2)
    label = y_test[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

def main():
    x = np.load('x.npy') / 255
    y = np.load('y.npy')
    x, y = shuffle(x, y)

    m, n, _ = x.shape
    x = x.reshape(m, n * n)
    y = y.reshape(m, 1)

    train_size = int(m * 0.8)

    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    w1 = np.random.rand(10, n * n)
    b1 = np.random.rand(10, 1)
    w2 = np.random.rand(10, 10)
    b2 = np.random.rand(10, 1)

    lr = 0.1
    epochs = 1000

    w1, b1, w2, b2 = gradient_descent(w1, w2, b1, b2, x_train, y_train, epochs, lr)
    test_prediction(5,w1,b1,w2,b2,x_test,y_test)

if __name__ == '__main__':
    main()
