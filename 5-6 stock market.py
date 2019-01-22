#%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
安裝 pandas-datareader¶
注意要安裝 pandas-datareader 套件

conda install pandas-datareader
如果之前裝過, 但有陣子沒用, 請更新

conda update pandas-datareader
"""
import pandas_datareader.data as web

df = web.DataReader("AAPL", "yahoo", start="2012-9-1", end="2017-8-31")

df.head()

P = df["Adj Close"]

P.head()

P.plot()


"""​
計算報酬率
𝑃𝑡−𝑃𝑡−1𝑃𝑡−1
"""
r = P.diff()/P

r[-100:].plot()

#移動平均

P.rolling(window=20).mean()

P.plot()
P.rolling(window=20).mean().plot()
P.rolling(window=60).mean().plot()