%matplotlib inline
# version 2. test gitHub
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# test git
df = pd.read_csv("grades.csv")
df.國文.mean()
df.國文.std()
df.describe()
df.corr()
df.國文.corr(df.數學)
df["總級分"] = df[["國文", "英文", "數學", "社會", "自然"]].sum(1)
df["主科"] = df.數學*1.5 + df.英文
df.head()
df.sort_values(by="總級分", ascending=False).head(20)
df.sort_values(by=["主科", "總級分"], ascending=False).head(20)
# version 1.0