import numpy as np


def q_01():
    A = np.random.rand(5, 5)
    minimum, maximum = A.min(), A.max()
    print((A - minimum) / (maximum - minimum))
    print("")


def q_02():
    value = 5
    B = np.random.rand(30) * 10
    print(B)
    print(value)  
    print(B[np.abs(B - value).argmin()])
    print("")


def q_03():
    n = 5
    a = np.random.randint(1, 100, 30)
    print(a)
    print(a[np.argsort(a)[-n:]])
    print("")


def q_04():
    X = np.random.rand(10, 4)
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    Z = (X - mu) / sigma
    print(Z)
    print("")


if __name__ == '__main__':
    q_01()
    q_02()
    q_03()
    q_04()
