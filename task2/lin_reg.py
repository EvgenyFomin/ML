__author__ = 'wolfram'

import numpy as np
import matplotlib.pyplot as plt
import pickle

def lin_reg(x, y):
    matrixA = np.array([[1 for i in range(0, len(x))], [j for j in x]]).transpose()
    b = np.dot(matrixA.transpose(), matrixA)
    b = np.linalg.inv(b)
    b = np.dot(b, matrixA.transpose())
    b = np.dot(b, y)
    ystar = np.dot(matrixA, b)
    sse = np.dot((y - ystar).transpose(), y - ystar)
    return b[0, 0], b[1, 0], sse

fig = plt.figure()

ax1 = fig.add_subplot(311)
ax1.set_title('dataset_1')
f = open('task2_dataset_1.txt', 'rb')
x, y = pickle.load(f, encoding='ISO-8859-1')
ax1.scatter(x, y)
b = [i for i in lin_reg(x, y)]
y = [b[0] + b[1]*i for i in x]
ax1.plot(x, y)
print(b[2])

ax2 = fig.add_subplot(312)
ax2.set_title(u'dataset_2')
f = open('task2_dataset_2.txt', 'rb')
x, y = pickle.load(f, encoding='ISO-8859-1')
ax2.scatter(x, y)
b = [i for i in lin_reg(x, y)]
y = [b[0] + b[1]*i for i in x]
ax2.plot(x, y)
print(b[2])

ax3 = fig.add_subplot(313)
ax3.set_title(u'dataset_3')
f = open('task2_dataset_3.txt', 'rb')
x, y = pickle.load(f, encoding='ISO-8859-1')
ax3.scatter(x, y)
b = [i for i in lin_reg(x, y)]
y = [b[0] + b[1]*i for i in x]
ax3.plot(x, y)
print(b[2])

plt.tight_layout(h_pad = 1)
plt.show()

f.close()