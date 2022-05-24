from matplotlib.pyplot import axis
import numpy as np
import pandas as pd



#data = np.loadtxt('logistic_regression/breast-cancer-wisconsin.data', delimiter=',', dtype=str)
data = pd.read_csv('logistic_regression/breast-cancer-wisconsin.data', delimiter=',', header=None)

data = data.iloc[:, 1:]
data = data[data != '?']
data = data.dropna( axis='rows')

data.loc[data[10] == 4, 10] = 1
data.loc[data[10] == 2, 10] = 0

#print(data.iloc[23])
print(data)

data.to_csv('logistic_regression/bcw/breast-cancer-wisconsin.csv',index=False, header= False)