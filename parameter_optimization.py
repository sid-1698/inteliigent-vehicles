from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import numpy as np
from pathlib import Path
from assignment_features import assignment_features
from sklearn.model_selection import GridSearchCV

pca  = PCA()
clf = SVC(kernel = "rbf", gamma = "scale")

''' C - Penalty parameter
    Increasing C => Overfitting. (or assuming that the training sample has all possible extreme cases).
'''

pipe = Pipeline(steps=[("PCA", pca), ("Classifier", clf)])
param_grid = {"PCA__n_components" : [15,20,25,30,35,40,45],
              "Classifier__C" : [0.001, 0.005, 0.01, 0.02, 0.04, 0.1, 0.5, 1, 2, 5, 10]}

if Path("assignment_features.npy").is_file():
    data = np.load("assignment_features.npy", allow_pickle=True).item()
else:
    data = assignment_features()

X_train = data["features"]["hog"]["train"]
y_train = data["y_train"]

grid = GridSearchCV(estimator=pipe, param_grid=param_grid, refit=True, verbose=3)
grid.fit(X_train, y_train)
print (grid.best_estimator_)
print(grid.cv_results_["mean_test_score"])