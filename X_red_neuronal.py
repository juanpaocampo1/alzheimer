import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(theta, X, y):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    cost = (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

def shuffle(x, y):
    r_indexes = np.arange(len(x))
    np.random.shuffle(r_indexes)
    return x[r_indexes], y[r_indexes]

def gradient_descent(X, y, theta, lr, epochs, batch_size):
    m = len(y)
    cost_history = []
    
    for _ in range(epochs):
        for batch_start in range(0, m, batch_size):
            batch_end = batch_start + batch_size
            x_batch = X[batch_start:batch_end]
            y_batch = y[batch_start:batch_end]
            
            h = sigmoid(np.dot(x_batch, theta))
            gradient = np.dot(x_batch.T, (h - y_batch)) / batch_size
            theta -= lr * gradient
            
            cost = compute_cost(theta, X, y)
            cost_history.append(cost)
    
    return theta, cost_history

def main():
    x = np.load('x.npy') / 255
    y = np.load('y.npy')
    x, y = shuffle(x, y)

    m, n, _ = x.shape
    x = x.reshape(m, n * n)
    x = np.hstack((np.ones((m, 1)), x))
    y = y.reshape(m, 1)

    train_size = int(m * 0.8)
    test_size = int(m * 0.1)

    x_train, x_test = x[:train_size], x[train_size + test_size:]
    y_train, y_test = y[:train_size], y[train_size + test_size:]

    theta = np.zeros((n * n + 1, 1))
    lr = 0.1
    epochs = 1000
    batch_size = 1000

    trained_theta, cost_history = gradient_descent(x_train, y_train, theta, lr, epochs, batch_size)

    predictions = sigmoid(np.dot(x_test, trained_theta))
    predicted_labels = (predictions >= 0.5).astype(int)
    accuracy = np.mean(predicted_labels == y_test) * 100

    print("Accuracy:", accuracy)

    for i in range(10):  # Primeros 10 ejemplos y sus predicciones
        print("Número real:", y_test[i], "Número predecido:", predicted_labels[i])

    plt.plot(cost_history)
    plt.xlabel('Iteraciones')
    plt.ylabel('Costo')
    plt.title('Historial de Costos')
    plt.show()

if __name__ == '__main__':
    main()