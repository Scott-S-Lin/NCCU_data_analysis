%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(np.random.randn(5,3),
                 index=list(range(1,6)),
                 columns=list("ABC"))

df
df[df.B>0]
df[df.B>0]["C"] = 0

df

df.loc[2:3, "B":"C"]
df.loc[2, "B"]
df.loc[2, "B"] = 1
df.loc[df.B>0, "C"] = 0
df
