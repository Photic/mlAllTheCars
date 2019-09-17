import R_functions as rf
import pandas as pd
import numpy as np
import statistics as st


mydata = pd.read_csv('./car.csv', header = None)
numpyArray = mydata

print(st.mode(numpyArray[6]))
