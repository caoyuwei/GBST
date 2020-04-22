import numpy as np
import gbst
from gbst.sklearn import gbstModel
from gbst.metrics import evalauc
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sksurv.datasets import load_flchain
from sksurv.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


x, y = load_flchain()

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=0)
num_columns = ['age', 'creatinine', 'kappa', 'lambda']
imputer = SimpleImputer().fit(train_x.loc[:, num_columns])
train_x = imputer.transform(train_x.loc[:, num_columns])
test_x = imputer.transform(test_x.loc[:, num_columns])

y_events = train_y[train_y['death']]
train_min, train_max = y_events["futime"].min(), y_events["futime"].max()

y_events = test_y[test_y['death']]
test_min, test_max = y_events["futime"].min(), y_events["futime"].max()
print(train_min,train_max,test_min,test_max)
assert train_min <= test_min < test_max < train_max, \
    "time range or test data is not within time range of training data."

times = np.percentile(y["futime"], np.linspace(5, 81, 15))
print(times)
train_y = train_y["futime"] // 200
test_y = test_y["futime"] // 200

classifier = gbstModel(n_estimators=100, num_class = 27)
# n_estimators = num_boost_rounds.
# num_class is #total_timewindows + 1
classifier.fit(X=train_x, y=train_y, eval_set=[(train_x, train_y), (test_x, test_y)], verbose=True)
# data format:
# X: numpy array. [dataset_size, num_features]
# y: 1-d numpy int array. Each index means "the number of timewindows that this sample survives". [dataset_size]

a = classifier.predict_hazard(data=test_x)
print(a)
# the output of predict_hazard() is h(t). Accumulated surviving probability should be calculated manually.

"""
Plot the tree structure/feature importance.
Graphviz or matplotlib needs to be installed.
"""

# from gbst.plotting import plot_tree,plot_importance
# import matplotlib.pyplot as plt
# plot_tree(classifier.get_booster())
# plot_importance(classifier.get_booster()) 
# plt.show()

"""
Run grid search cv.
"""
classifier_cv = gbstModel(num_class = 26)
param_grid = {"n_estimators": range(40,140,10), "max_depth": range(3,5,1)}
GridCV = GridSearchCV(classifier, param_grid, cv=None, verbose=3)
GridCV.fit(train_x, train_y)
print(GridCV.best_params_)
print(GridCV.best_score_)

