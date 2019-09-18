import R_functions as rf
import pandas as pd
import numpy as np
import statistics as st


mydata = pd.read_csv('./car.csv', header = None)
numpyArray = mydata

print(st.mode(numpyArray[6]))


index = 0

# Buying price
for i in range(1728):
    if(numpyArray[index][i] == 'vhigh'):
        numpyArray[index][i] = 1
    if(numpyArray[index][i] == 'high'):
        numpyArray[index][i] = 2
    if(numpyArray[index][i] == 'med'):
        numpyArray[index][i] = 3
    if(numpyArray[index][i] == 'low'):
        numpyArray[index][i] = 4

# Maintenance price

index = 1

for i in range(1728):
    if(numpyArray[index][i] == 'vhigh'):
        numpyArray[index][i] = 1
    if(numpyArray[index][i] == 'high'):
        numpyArray[index][i] = 2
    if(numpyArray[index][i] == 'med'):
        numpyArray[index][i] = 3
    if(numpyArray[index][i] == 'low'):
        numpyArray[index][i] = 4

# Doors

index = 2

for i in range(1728):
    if(numpyArray[index][i] == '2'):
        numpyArray[index][i] = 1
    if(numpyArray[index][i] == '3'):
        numpyArray[index][i] = 2
    if(numpyArray[index][i] == '4'):
        numpyArray[index][i] = 3
    if(numpyArray[index][i] == '5more'):
        numpyArray[index][i] = 4

# Doors

index = 3

for i in range(1728):
    if(numpyArray[index][i] == '2'):
        numpyArray[index][i] = 1
    if(numpyArray[index][i] == '4'):
        numpyArray[index][i] = 2
    if(numpyArray[index][i] == 'more'):
        numpyArray[index][i] = 3


# Luggage Boot

index = 4

for i in range(1728):
    if(numpyArray[index][i] == 'small'):
        numpyArray[index][i] = 1
    if(numpyArray[index][i] == 'med'):
        numpyArray[index][i] = 2
    if(numpyArray[index][i] == 'big'):
        numpyArray[index][i] = 3

# Safety

index = 5

for i in range(1728):
    if(numpyArray[index][i] == 'low'):
        numpyArray[index][i] = 1
    if(numpyArray[index][i] == 'med'):
        numpyArray[index][i] = 2
    if(numpyArray[index][i] == 'high'):
        numpyArray[index][i] = 3

# Car Evaluation

index = 6

for i in range(1728):
    if(numpyArray[index][i] == 'unacc'):
        numpyArray[index][i] = 1
    if(numpyArray[index][i] == 'acc'):
        numpyArray[index][i] = 2
    if(numpyArray[index][i] == 'good'):
        numpyArray[index][i] = 3
    if(numpyArray[index][i] == 'vgood'):
        numpyArray[index][i] = 4


for x in range(7):
    for y in range(1728):
        numpyArray[x][y] = float(numpyArray[x][y])

for x in range(7):
    mean = np.mean(numpyArray[x])
    var = rf.var(numpyArray[x])
    print("mean", mean)
    print("var", var)
    #for y in range(1728):
     #   (numpyArray[x][y]) = (numpyArray[x][y]) - mean
      #  (numpyArray[x][y]) = (numpyArray[x][y])/var



import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.scatter(numpyArray[5], numpyArray[6])
plt.show() # Depending on whether you use IPython or interactive mode, etc.

print(numpyArray)