import numpy as np

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def d_sigmoid(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))

# calculate the mean squared error
# y = actual, y_hat = predicted
def mse(y, y_hat):
    return np.multiply(0.5, (y - y_hat))

def cost(x, y, params):
    m = len(y[0])
    y_hat = forward_prop(x, params)[2]
    logprobs = np.multiply(np.log(y_hat), y) + np.multiply((1 - y), np.log(1 - y_hat))
    return np.sum(logprobs) / -m

def forward_prop(x, params):
    w1, w2, b1, b2 = params
    z1 = np.dot(w1, x) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)
    return a1, z2, a2

def predict(x, params):
    return np.round(forward_prop(x, params)[2])

def accuracy(x, y, params):
    m = float(x.shape[1])
    pred = predict(x, params)
    acc = np.abs(y - pred)
    return (m - np.sum(acc)) / m

#w1 = (h_dim, x_dim)
#w2 = (y_dim, h_dim)
#b1 = (h_dim, 1)
#b2 = (y_dim, 1)
def backprop(x, y, learning_rate, params):
    w1,w2,b1,b2 = params
    #m = float(y.shape[1])
    m = float(x.shape[1])
    a1, z2, a2 = forward_prop(x, params)
    dw2 = np.dot((a2-y), a1.T)
    db2 = np.sum((a2-y), axis=1, keepdims=True)
    # round 2
    dCda1 = np.dot(w2.T, (a2 - y))
    da1dz1 = (1.0 - np.power(a1, 2))
    dz1dw1 = x.T
    dw1 = np.dot(np.multiply(dCda1, da1dz1), dz1dw1)
    db1 = np.sum(np.multiply(dCda1, da1dz1), axis=1, keepdims=True)
    w1 = w1 - ((learning_rate / m) * dw1)
    w2 = w2 - ((learning_rate / m) * dw2)
    b1 = b1 - ((learning_rate / m) * db1)
    b2 = b2 - ((learning_rate / m) * db2)
    print('Cost: ', cost(x, y, params))
    print('Accuracy: ', accuracy(x, y, params))
    return [w1, w2, b1, b2]

def plot_solution(x,y,params):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
    fig, ax = plt.subplots()
    Z = predict(np.c_[xx.ravel(), yy.ravel()].T,params)
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    ax.axis('off')
    plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);
    ax.set_title('The solution set')
    plt.show()
