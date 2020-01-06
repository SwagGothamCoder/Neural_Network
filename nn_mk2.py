import numpy as np

class Neural_Network:
    def __init__(self, x_dim, h_dim, y_dim):
        print(x_dim, y_dim, h_dim)
        print(type(x_dim), type(y_dim), type(h_dim))
        self.w1 = np.random.randn(h_dim, x_dim)
        self.w2 = np.random.randn(y_dim, h_dim)
        self.b1 = np.zeros(shape=(h_dim, 1))
        self.b2 = np.zeros(shape=(y_dim, 1))
    def __str__(self):
        return 'Neural_Network with:\n' + str(self.w1.shape[1]) + ' input nodes, \n' + str(self.w1.shape[0]) + ' hidden nodes, \n' + 'and ' + str(self.b2.shape[0]) + ' output nodes.'
    def __sigmoid(self, z):
        return 1.0 / (1 + np.exp(-z))
    def __d_sigmoid(self, z):
        return np.multiply(self.__sigmoid(z), (1 - self.__sigmoid(z)))
    def __forward_prop(self, x):
        z1 = np.dot(self.w1, x) + self.b1
        a1 = np.tanh(z1)
        z2 = np.dot(self.w2, a1) + self.b2
        a2 = self.__sigmoid(z2)
        return a1, z2, a2
    # y = actual, y_hat = predicted
    def cost(self, x, y):
        m = len(y[0])
        y_hat = self.forward_prop(x)[2]
        logprobs = np.multiply(np.log(y_hat), y) + np.multiply((1 - y), np.log(1 - y_hat))
        return np.sum(logprobs) / -m
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
    #self.w1 = (h_dim, x_dim)
    #self.w2 = (y_dim, h_dim)
    #self.b1 = (h_dim, 1)
    #self.b2 = (y_dim, 1)
    def backprop(self, x, y, learning_rate):
        #m = float(y.shape[1])
        m = float(x.shape[1])
        a1, z2, a2 = self.forward_prop(x)
        dw2 = np.dot((a2-y), a1.T)
        db2 = np.sum((a2-y), axis=1, keepdims=True)
        # round 2
        dCda1 = np.dot(self.w2.T, (a2 - y))
        da1dz1 = (1.0 - np.power(a1, 2))
        dz1dw1 = x.T
        dw1 = np.dot(np.multiply(dCda1, da1dz1), dz1dw1)
        db1 = np.sum(np.multiply(dCda1, da1dz1), axis=1, keepdims=True)
        self.w1 = self.w1 - ((learning_rate / m) * dw1)
        self.w2 = self.w2 - ((learning_rate / m) * dw2)
        self.b1 = self.b1 - ((learning_rate / m) * db1)
        self.b2 = self.b2 - ((learning_rate / m) * db2)
        #print('Cost: ', self.cost(x, y))
        #print('Accuracy: ', self.accuracy(x, y))
        return [self.w1, self.w2, self.b1, self.b2]
