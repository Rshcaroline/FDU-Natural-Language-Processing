# This file is used to check whether your grad is right or not.

import numpy as np
import random
from scipy.special import expit


def sigmoid(x):
    return expit(x)


def sigmoid_grad(f):
    return f - f * f


def gradcheck_naive(f, x):
    """
    Gradient check for a function f
    - f should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
    """

    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x)    # Evaluate function value at original point

    y = np.copy(x)
    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        ### make sure you call random.setstate(rndstate) before calling f(x) each time, this will make it
        ### possible to test cost functions with built in randomness later
        reldiff = 1.0
        for negative_log_h in range(2, 22):
            h = 0.5 ** negative_log_h
            y[ix] = x[ix] + h
            random.setstate(rndstate)
            fy, _ = f(y)
            y[ix] = x[ix]
            numgrad = (fy - fx) / h
            # print(fx)
            # print(fy)
            if fx != fy:
                reldiff = min(reldiff, abs(numgrad - grad[ix]) / max((1.0, abs(numgrad), abs(grad[ix]))))
        if reldiff > 1e-5:
            print("Gradient check failed.")
            print("First gradient error found at index %s" % str(ix))
            print(grad[ix])
            print(numgrad)
            print("Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad))
            return
        it.iternext()  # Step to next dimension

    print("Gradient check passed!")


def sanity_check():
    """
    Some basic sanity checks.
    """
    quad_and_grad = lambda x: (np.sum(x ** 2), x * 2)
    print("Running sanity checks...")
    gradcheck_naive(quad_and_grad, np.array(123.456))  # scalar test
    gradcheck_naive(quad_and_grad, np.random.randn(3, ))  # 1-D test
    gradcheck_naive(quad_and_grad, np.random.randn(4, 5))  # 2-D test
    print()

if __name__ == "__main__":
    sanity_check()