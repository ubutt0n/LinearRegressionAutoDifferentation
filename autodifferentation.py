import numpy as np
from collections import defaultdict

class Variable:
    def __init__(self, value, local_gradients=()):
        self.value = value
        self.local_gradients = local_gradients
    
    def __add__(self, other):
        return add(self, other)
    
    def __mul__(self, other):
        return mul(self, other)
    
    def __sub__(self, other):
        return add(self, neg(other))
    
    def __truediv__(self, other):
        return mul(self, inv(other))

def add(a, b):
    value = to_np_vals(a) + to_np_vals(b)
    local_gradients = (
        (a, 1),
        (b, 1)
    )
    return Variable(value, local_gradients)

def mul(a, b):
    value = to_np_vals(a) * to_np_vals(b)
    local_gradients = (
        (a, to_np_vals(b)),
        (b, to_np_vals(a))
    )
    return Variable(value, local_gradients)

def meanv(a):
    value = 1. / a.value
    local_gradients = (
        (a, -1 / a.value**2)
    )
    return Variable(value, local_gradients)

def inv(a):
    value = 1. / a.value
    local_gradients = (
        (a, -1 / a.value**2),
    )
    return Variable(value, local_gradients) 

def neg(a):
    value = -1 * to_np_vals(a)
    local_gradients = (
        (a, -1),
    )
    return Variable(value, local_gradients)

def exp(a):
    value = np.exp(a.value)
    local_gradients = (
        (a, value),
    )
    return Variable(value, local_gradients)

def log(a):
    value = np.log(a.value)
    local_gradients = (
        (a, 1. / a.value),
    )
    return Variable(value, local_gradients)

def get_grad(variable):
    gradient = defaultdict(lambda: 0)
    def compute_grad(variable, path_value):
        for var, local_grad in variable.local_gradients:
            path_to_child = path_value * local_grad
            gradient[var] += path_to_child
            compute_grad(var, path_to_child)
    compute_grad(variable, path_value=1)

    return gradient

to_np = np.vectorize(lambda x: Variable(x))
to_np_vals = np.vectorize(lambda variable: variable.value)