import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb

df = pd.read_csv('https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv').fillna(0)

full, test = train_test_split(df, test_size=0.2, random_state=1)
train, val = train_test_split(full, test_size=0.25, random_state=1)
y_train = train.pop('fuel_efficiency_mpg')
y_val = val.pop('fuel_efficiency_mpg')

dv = DictVectorizer(sparse=True)
X_train = dv.fit_transform(train.to_dict('records'))
X_val = dv.transform(val.to_dict('records'))
names = list(dv.get_feature_names_out())


def rf_rmse(n, depth=None):
    rf = RandomForestRegressor(n_estimators=n, max_depth=depth, random_state=1, n_jobs=-1)
    rf.fit(X_train, y_train)
    return root_mean_squared_error(y_val, rf.predict(X_val)), rf


# Q1
dt = DecisionTreeRegressor(max_depth=1).fit(X_train, y_train)
print('Q1:\n', export_text(dt, feature_names=names))

# Q2
print('Q2:', rf_rmse(10)[0])

# Q3
for n in [10, 50, 100, 150]:
    print('Q3:', n, round(rf_rmse(n)[0], 3))

# Q4
for d in [10, 15, 20, 25]:
    scores = [rf_rmse(n, d)[0] for n in [10, 50, 100, 150]]
    print('Q4:', d, sum(scores) / len(scores))

# Q5
rf = rf_rmse(10, 20)[1]
print('Q5:\n', pd.Series(rf.feature_importances_, index=names).sort_values(ascending=False).head())

# Q6
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=names)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=names)
for eta in [0.3, 0.1]:
    params = {'eta': eta, 'max_depth': 6, 'min_child_weight': 1,
              'objective': 'reg:squarederror', 'nthread': 8, 'seed': 1}
    model = xgb.train(params, dtrain, num_boost_round=100)
    print('Q6:', eta, root_mean_squared_error(y_val, model.predict(dval)))
