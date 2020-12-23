from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import numpy as np
from pathlib import Path
from assignment_features import assignment_features
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

if Path("assignment_features.npy").is_file():
    data = np.load("assignment_features.npy", allow_pickle=True).item()
else:
    data = assignment_features()
    
    
pca  = PCA(n_components=(35))
clf = SVC(kernel = "rbf", gamma = "scale", C=5)
pipe = Pipeline(steps=[("PCA", pca), ("Classifier", clf)])
_,(ax1,ax2)=plt.subplots(2,1,figsize=(15,15))
pipe.fit(data["features"]["hog"]["train"], data["y_train"])
plot_confusion_matrix(pipe, data["features"]["hog"]["test"], data["y_test"], cmap="Blues", ax=ax1)
pipe.fit(data["features"]["cnn"]["train"], data["y_train"])
plot_confusion_matrix(pipe, data["features"]["cnn"]["test"], data["y_test"], cmap="Blues", ax=ax2)
plt.show()

