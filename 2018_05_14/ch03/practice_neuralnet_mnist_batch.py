# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # ファイルをインポートするための設定
import numpy as np
import pickle # pickle fileを使用するための設定
from dataset.mnist import load_mnist
from functions import sigmoid, softmax

# データを取得する
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

#訓練済みの辞書型重みデータを取り入れる
def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

#3層のネットワークによる計算
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()
batch_size = 100
accuracy_cnt = 0 #正解に当たった数を数える

#バッチ処理によって精度を求める







print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
