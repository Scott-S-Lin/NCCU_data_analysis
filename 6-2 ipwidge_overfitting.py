%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

from ipywidgets import interact

x = np.linspace(0, 5, 50)
y = 1.2*x + 0.8 + 0.6*np.random.randn(50)

plt.scatter(x,y)
plt.plot(x, 1.2*x + 0.8, 'r')

X = np.linspace(0, 5, 1000)

def my_fit(n):
    Y = 4*np.sin(n*X) + 4
    plt.scatter(x, y)
    plt.plot(X, Y, 'r')
    plt.show()

my_fit(5)

interact(my_fit, n=(1, 500))