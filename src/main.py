import R_functions as rf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as st
from matplotlib.patches import Polygon
from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show
import scipy.stats
import seaborn as sns
from scipy.linalg import svd

# First we start off by importing our data set using pandas.

df = pd.read_csv('./car.csv', header = None)

# On the website it said there were no missing values, but we just check for good measure.
print("\nCalling mydata.info() to see if there are any missing values in the data set.")
df.info()

# The data came with no column names, so I'm just going to write them in.
# Purchase price, maintenance price, number of doors, person capacity, size of luggage boot, safety, overall evaluation(classification).
df.columns = ['price', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

# Looking at head & tail of the data set
print("Printing head and tail for the data set")
print(df)

# Looking at distribution of records for each attribute.
for i in df.columns:
    print(df[i].value_counts())
    print()

# To do any kind of summary statistics, we have to convert these categorical / non-numerical values into integers.
# If these non-numerical values were representing a ranking system, we would use out-of-K coding, but since these words clearly represent a hierachy it is easy to just switch them out for integers.
# Thankfully the data set is sorted in such a way that it is easy to see, that the lower the price, the better the evaluation, and the more comfort and tech, the better.
# The different attribute values have already been covered in the report, but the distribution of them could be interesting to look at.

# https://www.geeksforgeeks.org/replacing-strings-with-numbers-in-python-for-data-analysis/
price_label = {"vhigh" : 0, "high" : 1, "med" : 2, "low" : 3}
lug_label = {"small" : 0, "med" : 1, "big" : 2}
safety_label = {"low" : 0, "med" : 1, "high" : 2}
doors_label = {"2" : 0, "3" : 1, "4" : 2, "5more" : 3}
persons_label = {"2" : 0, "4" : 1, "more" : 2}
class_label = {"unacc" : 0, "acc" : 1, "good" : 2, "vgood" : 3}

df.price = [price_label[item] for item in df.price]
df.maint = [price_label[item] for item in df.maint]
df.lug_boot = [lug_label[item] for item in df.lug_boot]
df.safety = [safety_label[item] for item in df.safety]
df.doors = [doors_label[item] for item in df.doors]
df.persons = [persons_label[item] for item in df.persons]
df['class'] = [class_label[item] for item in df['class']]

print(df)

# This shows that everything besides class is equally distributed.
# One thing in this data set that differs from the data we have used in the exercises so far,
# is that all the values are non-numeric so we have to

# Ex 1_5_1
raw_data = df.get_values()
cols = range(0, 7)
X = raw_data[:, cols]
attributeNames = np.asarray(df.columns[cols])
classLabels = raw_data[:,-1] # -1 takes the last column
classNames = np.unique(classLabels)
classDict = dict(zip(classNames,range(len(classNames))))
y = np.array([classDict[cl] for cl in classLabels])
print(y)
N, M = X.shape
C = 4 # because there are 4 different class labels.
print(X)

# Doing some summary statistics like mean, variance, standard deviation, median, quantiles etc.

print("Mean with np")
print(X.mean(axis=0))
print("Standard Deviation with np")
print(np.std(X, axis=0))
print("Variance with np")
print(np.var(X, axis=0))
print("Quantiles with np")
print("Min", np.min(X, axis=0))
print("Q25", np.quantile(X, 0.25, axis=0))
print("Q50/Median", np.quantile(X, 0.50, axis=0))
print("Q75", np.quantile(X, 0.75, axis=0))
print("Max", np.max(X, axis=0))

# We saw previously that all the data the was equally distributed beside the class label, so they are not
# very interesting to plot alone.

# Histogram of price
plt.hist(df['price'])
plt.show()
# Histogram of Class alone.
plt.hist(df['class'])
plt.show()

# Below code snippet comes from GitHub, we had trouble visualizing the data so we wanted to see how others had,
# created meaningful plots to show the distribution of attributes amongst classifications.
# https://github.com/sonarsushant/Car-Evaluation-Dataset-Classification/blob/master/Car%20Evaluation%20Dataset.ipynb

for i in df.columns[:-1]:
    plt.figure(figsize=(12,6))
    plt.title("For feature '%s'"%i)
    sns.countplot(df[i],hue=df['class'])
    plt.savefig('ClassVs%s'%i)


# Subtract mean value from data (ex. 2.1.3)
Y = X - np.ones((N,1))*X.mean(axis=0)

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.savefig('VarianceExplainedByPCA')
plt.show()

# PCA by computing SVD of Y
U,S,Vh = svd(Y,full_matrices=False)

# of the vector V. So, for us to obtain the correct V, we transpose:
V = Vh.T    

# Project the centered data onto principal component space
Z = Y @ V

# Indices of the principal components to be plotted We plot each attribute up against the other
j =6

# Plot PCA of the data

for i in range(7):
    f = figure()
    title('PCA Car Evaluation vs %s'%attributeNames[i])
    #Z = array(Z)
    for c in range(C):
    # select indices belonging to class c:
        class_mask = y==c
        plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
        legend(classNames)
        xlabel('PC{0}'.format(j+1))
        ylabel('PC{0}'.format(i+1))
    plt.savefig('PCA Car Evaluation vs %s'%attributeNames[i])
    # Output result to screen
show()