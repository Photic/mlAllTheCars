import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import scipy.stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('./car.csv', names = ['price', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])

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

df = StandardScaler().fit_transform(df)

pca = PCA(n_components=7)
principalComponents = pca.fit_transform(df)

print("PCA Ratio " + str(pca.explained_variance_ratio_))