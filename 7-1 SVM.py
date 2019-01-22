%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

x = np.array([[-3,2], [-6,5], [3,-4], [2,-8]])
y = np.array([1, 1, 2, 2])

plt.scatter([-3, -6, 3, 2], [2, 5, -4, -8], c=y)

x

x[2, 1]

x[:,0]

x[:,1]

plt.scatter(x[:,0], x[:,1], c=y)
###  X x Y
plt.scatter(x[:,0], x[:,1], s=50, c=y)

x

y

from sklearn.svm import SVC

clf = SVC()

clf.fit(x, y)

clf.predict([[-3,2]])
clf.predict(x)

clf.predict([[2.5,3]])

## Meshgrid

xx = [1,2,3,4]
yy = [5,6,7,8]

X, Y = np.meshgrid(xx,yy)
X
Y

X, Y = np.meshgrid(np.linspace(-6,3,30), np.linspace(-8,5,30))

X = X.ravel()
Y = Y.ravel()

plt.scatter(X, Y)

xx = [1,2,3,4]
yy = [5,6,7,8]

list(zip(xx,yy))

Z = clf.predict(list(zip(X,Y)))

plt.scatter(X, Y, s=50, c=Z)







