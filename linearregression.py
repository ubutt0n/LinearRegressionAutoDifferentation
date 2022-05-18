from autodifferentation import Variable, to_np, get_grad, to_np_vals
import numpy as np
import matplotlib.pyplot as plt

def update_w(w, grad, lrate):
    for _, weight in np.ndenumerate(w):
        w[0].value -= lrate * grad[weight]

def update_b(b, grad, lrate):
    b.value -= lrate * grad[b]

class LinearRegression:
    def __init__(self, lr=0.001):
        self.lr = lr

    def fit(self, x, y, ep=100):
        self.w = to_np(np.zeros(x.shape[1]))
        self.b = Variable(0)
        x = to_np(x)
        y = to_np(y)
        loss_arr = []
        
        for i in range(0, ep):
            cost = self.__loss(x, self.w, self.b, y)
            grad = get_grad(cost)
            update_w(self.w, grad, self.lr)
            update_b(self.b, grad, self.lr)
            loss_arr.append(to_np_vals(cost))
        plt.plot(loss_arr)

    def __loss(self, x, w, b, y):
        gay = Variable(1 / len(y))
        return gay * np.sum((y - (np.dot(x,w)+b)) * (y - (np.dot(x,w)+b)))
    
    def predict(self, x):
        x = to_np(x)
        pred = np.dot(x, self.w) + self.b
        return to_np_vals(pred)
