
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
from matplotlib import pyplot
import IOHandler
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

# Model to predict the zone of a contraband based on the coordinates of the body parts.
# Input : Coordinates of Head, Hands, Groin and Contraband.
# Output : Zone where the contraband is.

# Load data

custom_dataset  = np.loadtxt(IOHandler.CUSTOM_PREDICTED_DATA, delimiter=",")
labeled_dataset = np.loadtxt(IOHandler.LABELED_PREDICTED_DATA, delimiter=",")
dataset = np.concatenate((custom_dataset, labeled_dataset), axis=0)
#dataset = custom_dataset
#dataset = pd.read_csv(IOHandler.ZONE_PRED_FILE)

# split data into X and y
ncols = dataset.shape[1] - 1
X = dataset[:,0:ncols]
Y = dataset[:,ncols]

# split data into train and test sets
seed = 7
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
#
# # Make predictions
model = XGBClassifier(n_estimators=300, max_depth=4)
model.fit(X_train, y_train)

#make predictions for test data
y_pred = model.predict(X_test)
predictions = [int(value) for value in y_pred]

#Save the Model
IOHandler.saveZoneDetectorModel(model)

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

print(model.feature_importances_)
plot_importance(model)
pyplot.show()


#Hyperparameter Tuning
# model = XGBClassifier()
#
# n_estimators = [200, 300, 400, 500, 600]
# max_depth = [4, 6, 8, 10]
# param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)
#
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
# grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1)
# grid_result = grid_search.fit(X_train, y_train)
#
# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))

