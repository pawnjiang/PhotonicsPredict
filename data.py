import pandas as pd

# data = pd.read_csv("data2.csv",header=None)
# data.columns = ["data"]
# lam_val = 1550
# power = 1.3
# # data.insert(loc=len(data.columns),column='lambda',value = lam_val)
# # data.insert(loc=len(data.columns),column='power',value = power)
# data['lambda'] = lam_val
# data['power'] = power
# data.to_csv('data_2.csv',index=None)
# print(data)

data1 = pd.read_csv('data_1.csv',index_col=None)
data2 = pd.read_csv('data_2.csv',index_col=None)
data_all = pd.concat([data1,data2],ignore_index=True)
print(data_all.columns)

data_all.to_csv('data_final.csv',index=None)

# print(data_all)

data3 = pd.read_csv('data_final.csv',index_col=None)
print(data3.columns)