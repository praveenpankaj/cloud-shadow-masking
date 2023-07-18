# Import python libraries
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

# Import training data
x = pd.read_csv("./training_data/hollstein/x_full.csv")
y = pd.read_csv("./training_data/hollstein/y_full.csv")
# x_test = pd.read_csv('./training_data/over_under/x_test_10k.csv')
# y_test = pd.read_csv('./training_data/over_under/y_test_10k.csv')

# x_train = x_train_ou[['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8','B8A', 'B9', 'B10', 'B11', 'B12']]
# y_train = y_train_ou['cluster']
# x_test = x_test[['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8','B8A', 'B9', 'B10', 'B11', 'B12']]
# y_test = y_test['cluster']

print(x.shape, y.shape)
# print(x_test.shape, y_test.shape)

# sys.exit()

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 150, num = 20)]
max_features = ['auto', 'sqrt'] # Number of features to consider at every split
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None) # Maximum number of levels in tree
min_samples_split = [2, 5, 10] # Minimum number of samples required to split a node
min_samples_leaf = [1, 2, 4]  # Minimum number of samples required at each leaf node
bootstrap = [True, False]  # Method of selecting samples for training each tree

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print(random_grid)


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier(bootstrap=True,
                            class_weight=None,
                            criterion='gini',
                            max_depth=None,
                            max_features='auto',
                            max_leaf_nodes=None,
                            min_impurity_decrease=0.0,
                            min_impurity_split=None, # default=2
                            min_samples_leaf=1,
                            min_samples_split=2,
                            min_weight_fraction_leaf=0.0,
                            n_estimators=100,
                            n_jobs=1, # -1 means all processors
                            oob_score=False,
                            random_state=None,
                            verbose=0,
                            warm_start=False)

# Random search of parameters, using 3 fold cross validation 
# search across 30 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 30, cv = 7, verbose=20, random_state=42, n_jobs = 50)

# Fit the random search model
search = rf_random.fit(x, y)

# Save parameters of all models
res_df = pd.DataFrame(search.cv_results_)
res_df.to_csv('./models/rf_50t150_hyper_hollstein_full_13i_6o_model4.csv')

# Get Best model
best_model = search.best_estimator_

# # Estimate the accuracy from the best model
# y_pred_test = best_model.predict(x_test)
# acc = accuracy_score(y_test, y_pred_test)
# print("Overal Test Accuracy: ", acc)

# Save best RF Model
filename = './models/rf_50t150_hyper_hollstein_full_13i_6o_model4..sav'
pickle.dump(best_model, open(filename, 'wb'))

# # Test saved model by calculating accuracy again
# filename = './models/rf_uo10000_13i_25o_50to150_100it_7k.sav'
# loaded_model = pickle.load(open(filename, 'rb'))
# lm_result = loaded_model.score(x_test, y_test)
# print("Overal Test Accuracy: ", lm_result)
