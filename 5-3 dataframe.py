%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

mydata = np.random.randn(4,3)
mydata
list("ABCDE")
list("¥Ò¤A¤þ¤B")
df1 = pd.DataFrame(mydata, columns=list("ABC"))
df1
df2 = pd.DataFrame(np.random.randn(3,3), columns=list("ABC"))
df2
df3 = pd.concat([df1, df2], axis=0)
df3
df3.index = range(7)
df3
df4 = pd.concat([df1, df2], axis=1)
df4
