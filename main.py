import pandas as pd
import numpy as np

url = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv"
df = pd.read_csv(url)

print(pd.__version__)                                    # Q1
print(len(df))                                        # Q2
print(df.fuel_type.nunique())                             # Q3
print((df.isnull().sum() > 0).sum())        # Q4
print(df[df.origin == "Asia"].fuel_efficiency_mpg.max()) # Q5

median_before = df.horsepower.median()
mode_hp = df.horsepower.mode()[0]
median_after = df.horsepower.fillna(mode_hp).median()     # Q6
print(median_before, median_after)

asia = df[df.origin == "Asia"]

'''
asia = df[df.origin == "Asia"]
asia_subset = asia[["vehicle_weight", "model_year"]].iloc[:7]
asia_subset

X = asia_subset.values
X

XTX = X.T.dot(X)
XTX

XTX_inv = np.linalg.inv(XTX)
XTX_inv

y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

w = XTX_inv.dot(X.T).dot(y)
w

w.sum()
'''
X = asia[["vehicle_weight", "model_year"]].iloc[:7].values
XTX = X.T.dot(X)
w = np.linalg.inv(XTX).dot(X.T).dot(np.array([1100,1300,800,900,1000,1100,1200]))
print(w.sum())                                            # Q7