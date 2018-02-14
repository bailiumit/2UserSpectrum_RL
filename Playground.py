import numpy as np


A = np.array([[1, 0, 1]])
B = np.array([[1, 0, 0], [1, 2, 3], [4, 5, 6]])
C = B[np.nonzero(A>0)[1], :]
D = np.nonzero(A>0)

print(C)

# predY = np.array([[0.3, 0.7], [0.6, 0.4]])
# trueY = np.array([[0, 3], [1, 10]])
# s = np.sum(
#     np.log(np.abs(trueY[:, 0] - predY[:, 0])) * trueY[:, 1])

# print('Result: ', s)


# import keras.backend as K
# import numpy as np

# W1_ = np.array([[1, 2], [3, 4], [5, 60]])
# W2_ = np.array([[7], [8]])
# dW1 = np.array([[1, 2], [3, 4], [5, -60]])

# # W1 = K.variable(W1_)
# # W2 = K.variable(W2_)
# X = K.placeholder()

# W1 = K.variable(W1_+dW1)
# X_ = np.array([[-0.1, -0.2, 0.1]])

# Y = K.dot(K.relu(K.dot(X, W1)), W2)
# fn = K.function([X], K.gradients(Y, [W2]))

# X_ = np.array([[-0.1, -0.2, 0.1]])


# W1 = K.placeholder()
# W2 = K.placeholder()


# for i in range(1):
#     print(fn([X_]))



# model = {}
# model['A'] = np.random.randn(2, 3)
# B = np.random.randn(10)

# A = []
# H = []
# for i in range(5):
#     A.append(i)
#     H.append(np.random.randn(5, 1))

# epA = np.vstack(A)
# epH = np.vstack(H)

# # print(B.shape)
# # print(np.dot(epH.T, epA).ravel())

# H = 15
# D = 30

# W1 = np.random.randn(H, D)
# W2 = np.random.randn(H)

# x = np.ones((D, 1))

# h = np.dot(W1, x)
# logp = np.dot(W2, h)

# h2 = np.dot(W1, x)

# hs = []
# hs.append(h)
# hs.append(h2)

# print(np.hstack(hs))
