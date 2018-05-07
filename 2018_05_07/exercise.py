import numpy as np


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-1.1, -2.1]) # 重み
    b = 2.2 # バイアス
    value = np.sum(w * x) + b
    if value > 0:
        return 1
    else:
        return 0


def XOR(x1, x2):
    x3 = NAND(x1, x2)
    x4 = NAND(x1, x3)
    x5 = NAND(x2, x3)
    return NAND(x4, x5)


if __name__ == '__main__':

    for x in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        y = XOR(x[0], x[1])
        print(str(x) + " -> " + str(y))


