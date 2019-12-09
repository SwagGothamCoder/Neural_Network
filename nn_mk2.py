import numpy as np

class Neural_Network(object):
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size):
        w1 = np.random.randn(hidden_layer_size, input_layer_size)
        w2 = np.random.randn(output_layer_size, hidden_layer_size)
        b1 = np.zeros(shape=(hidden_layer_size, 1))
        b2 = np.zeros(shape=(output_layer_size, 1))
        self.params = [w1, w2, b1, b2]
    def sigmoid(self, z):
        return 1.0 / (1 + np.exp(-z))
    # y = actual, y_hat = predicted
    def cost(self, x, y):
        m = len(y[0])
        y_hat = self.forward_prop(x)[2]
        logprobs = np.multiply(np.log(y_hat), y) + np.multiply((1 - y), np.log(1 - y_hat))
        return np.sum(logprobs) / -m
    def forward_prop(self, x):
        w1, w2, b1, b2 = self.params
        z1 = np.dot(w1, x) + b1
        a1 = np.tanh(z1)
        z2 = np.dot(w2, a1) + b2
        a2 = self.sigmoid(z2)
        return a1, z2, a2
    def predict(self, x):
        results = self.forward_prop(x)[2]
        for i in range(len(results.T)):
            idx = np.where(results.T[i] == max(results.T[i]))[0][0]
            results.T[i] = np.zeros(shape=(len(results)))
            results.T[i][idx] = 1.0
        return results * np.round(results)
    def accuracy(self, x, y):
        m = float(x.shape[1])
        pred = self.predict(x)
        acc = np.abs(y - pred)
        return ((10*m) - np.sum(acc)) / (10*m)
    #w1 = (h_dim, x_dim)
    #w2 = (y_dim, h_dim)
    #b1 = (h_dim, 1)
    #b2 = (y_dim, 1)
    def backprop(self, x, y, learning_rate, batch_size, num_epochs):
        #m = float(y.shape[1])
        training_size = float(x.shape[1])
        m = batch_size
        num_batches = int(training_size / batch_size)
        x = np.array(np.split(x.T, num_batches))
        y = np.array(np.split(y.T, num_batches))
        for e in range(num_epochs):
            for i in range(num_batches):
                w1,w2,b1,b2 = self.params
                x_batch = x[i].T
                y_batch = y[i].T
                a1, z2, a2 = self.forward_prop(x_batch)
                dw2 = np.dot((a2-y_batch), a1.T)
                db2 = np.sum((a2-y_batch), axis=1, keepdims=True)
                # round 2
                dCda1 = np.dot(w2.T, (a2 - y_batch))
                da1dz1 = (1.0 - np.power(a1, 2))
                dz1dw1 = x_batch.T
                dw1 = np.dot(np.multiply(dCda1, da1dz1), dz1dw1)
                db1 = np.sum(np.multiply(dCda1, da1dz1), axis=1, keepdims=True)
                w1 = w1 - ((learning_rate / m) * dw1)
                w2 = w2 - ((learning_rate / m) * dw2)
                b1 = b1 - ((learning_rate / m) * db1)
                b2 = b2 - ((learning_rate / m) * db2)
                self.params = [w1, w2, b1, b2]
        #print('Cost: ', cost(x, y))
        #print('Accuracy: ', accuracy(x, y))
