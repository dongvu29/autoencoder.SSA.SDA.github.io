import pandas as pd
f=pd.read_csv("SSDPFlood_labels.csv")
keep_col = ['x']
new_f = f[keep_col]
new_f.drop(index='x')
new_f.to_csv("newFile.csv", index=False)
