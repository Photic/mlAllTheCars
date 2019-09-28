import R_functions as rf
import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import scipy.stats


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


plt.style.use('ggplot')
plt.scatter(numpyArray[5], numpyArray[6])
plt.show() # Depending on whether you use IPython or interactive mode, etc.

print(numpyArray)

columnTwo = numpyArray[2]

print("MEAN !!", rf.mean(columnTwo))

x_min = min(columnTwo)
x_max = max(columnTwo)

mean = rf.mean(columnTwo)
std = rf.sd(columnTwo)

x = np.linspace(x_min, x_max, 100)
y = scipy.stats.norm.pdf(x,mean,std)

plt.hist(numpyArray[6])
plt.plot(x,y, color='coral')

plt.grid()

plt.xlim(x_min,x_max)

plt.xlabel('x')
plt.ylabel('Normal Distribution of Doors')

plt.savefig("normal_distribution.png")
plt.show()


data = numpyArray[5]
print("MEAN 6", rf.mean(data))
plt.set_title = "Hello"
plt.boxplot(data)
plt.xticks([1], ["Doors Formatted"])
plt.show()