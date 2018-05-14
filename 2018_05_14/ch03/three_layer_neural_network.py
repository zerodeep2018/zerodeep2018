import numpy as np

#重みとバイアスのパラメーターを辞書型のnetwoekに格納する
def init_network():
	network = {}
	network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
	network['b1'] = np.array([0.1, 0.2, 0.3])
	network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
	network['b2'] = np.array([0.1, 0.2])
	network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
	network['b3'] = np.array([0.1, 0.2])

	return network

#ニューラルネットワーク順方向伝播の計算
def forward(network, x):
	W1, W2, W3 = network['W1'], network['W2'], network['W3']
	b1, b2, b3 = network['b1'], network['b2'], network['b3']

	a1 = np.dot(x, W1) + b1
	z1 = function_sigmoid(a1)
	a2 = np.dot(z1, W2) + b2
	z2 = function_sigmoid(a2)
	a3 = np.dot(z2, W3) + b3
	y = function_identify(a3)

	return y

#シグモイド関数
def function_sigmoid(x):
	return 1/(1+np.exp(-x))

#恒等関数
def function_identify(x):
	return x

if __name__ == '__main__':
#結果の出力
	network = init_network()
	x = np.array([1.0, 0.5])
	y = forward(network, x)
	print(y) 
