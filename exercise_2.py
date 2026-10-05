import pandas as pd
import numpy as np

url = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv"
cols = ["engine_displacement", "horsepower", "vehicle_weight", "model_year", "fuel_efficiency_mpg"]
df = pd.read_csv(url)[cols]

def prepare_X(df, fill_value):
    return df.drop(columns=["fuel_efficiency_mpg"]).fillna(fill_value).values

def train_linear_regression_reg(X, y, r=0.0):
    X = np.column_stack([np.ones(X.shape[0]), X])
    XTX = X.T.dot(X) + r * np.eye(X.shape[1])
    w_full = np.linalg.inv(XTX).dot(X.T).dot(y)
    return w_full[0], w_full[1:]

def rmse(y, y_pred):
    return np.sqrt(((y - y_pred) ** 2).mean())

def split(df, seed):
    n = len(df)
    n_val = n_test = int(n * 0.2)
    n_train = n - n_val - n_test
    np.random.seed(seed)
    idx = np.random.permutation(n)
    return (df.iloc[idx[:n_train]].reset_index(drop=True),
            df.iloc[idx[n_train:n_train+n_val]].reset_index(drop=True),
            df.iloc[idx[n_train+n_val:]].reset_index(drop=True))

df_train, df_val, df_test = split(df, 42)
y_train, y_val = df_train.fuel_efficiency_mpg.values, df_val.fuel_efficiency_mpg.values
mean_hp = df_train.horsepower.mean()

for fill, label in [(0, "0"), (mean_hp, "mean")]:
    w0, w = train_linear_regression_reg(prepare_X(df_train, fill), y_train)
    y_pred = w0 + prepare_X(df_val, fill).dot(w)
    print(label, round(rmse(y_val, y_pred), 3))

for r in [0, 0.01, 0.1, 1, 5, 10, 100]:
    w0, w = train_linear_regression_reg(prepare_X(df_train, 0), y_train, r)
    y_pred = w0 + prepare_X(df_val, 0).dot(w)
    print(r, round(rmse(y_val, y_pred), 4))

scores = []
for seed in range(10):
    dft, dfv, _ = split(df, seed)
    w0, w = train_linear_regression_reg(prepare_X(dft, 0), dft.fuel_efficiency_mpg.values)
    y_pred = w0 + prepare_X(dfv, 0).dot(w)
    scores.append(rmse(dfv.fuel_efficiency_mpg.values, y_pred))
print(round(np.std(scores), 3))

dft, dfv, dfte = split(df, 9)
df_full = pd.concat([dft, dfv]).reset_index(drop=True)
w0, w = train_linear_regression_reg(prepare_X(df_full, 0), df_full.fuel_efficiency_mpg.values, r=0.001)
y_pred = w0 + prepare_X(dfte, 0).dot(w)
print(rmse(dfte.fuel_efficiency_mpg.values, y_pred))