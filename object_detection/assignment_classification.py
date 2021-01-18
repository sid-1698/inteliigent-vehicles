from assignment_features import assignment_features
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as kNN
import sklearn
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import plot_confusion_matrix
from image_processing import increase_intensity
from tensorflow.keras.applications.mobilenet import MobileNet
from calculate_hog import calculate_hog
from calculate_cnn import calculate_cnn
from sklearn.decomposition import PCA  
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_roc_curve 

def assignment_classification(plot=False):

    ## Load the features (requires assignment 1 to be completed)
    if Path("assignment_features.npy").is_file():
        data = np.load("assignment_features.npy", allow_pickle=True).item()
    else:
        data = assignment_features()

    # Store classifiers in the data dict as well
    data["features"]["int"]["classifier"] = {}
    data["features"]["hog"]["classifier"] = {}
    data["features"]["cnn"]["classifier"] = {}

    ## Exercise 2.1: Train the svm on all three feature sets
    # Complete the code below
    # Save the resulting trained svm in the dictionary
    # Use kernel='rbf' and C=100
    
    kernel="rbf"
    C=2
    model = SVC(kernel=kernel, C=C)
    model.fit(data["features"]["int"]["train"], data["y_train"])
    data["features"]["int"]["classifier"]["svm"] = model

    model = SVC(kernel=kernel, C=C)
    model.fit(data["features"]["hog"]["train"], data["y_train"])
    data["features"]["hog"]["classifier"]["svm"] = model

    model = SVC(kernel=kernel, C=C)
    model.fit(data["features"]["cnn"]["train"], data["y_train"])
    data["features"]["cnn"]["classifier"]["svm"] = model

    # Check if the SVM is stored and can be used in later parts of the assignment
    if not "svm" in data["features"]["int"]["classifier"]:
        print("store the trained svm in data['features']['int']['classifier']['svm'], exiting..")
        sys.exit(-1)
    if not "svm" in data["features"]["hog"]["classifier"]:
        print("store the trained svm in data['features']['hog']['classifier']['svm'], exiting..")
        sys.exit(-1)
    if not "svm" in data["features"]["cnn"]["classifier"]:
        print("store the trained svm in data['features']['cnn']['classifier']['svm'], exiting..")
        sys.exit(-1)

    ## Exercise 2.2: Train the k-NN classifier on all three feature sets with k = 1
    # Complete the code below
    # Save the resulting trained classifier in the dictionary
    k=1
    model = kNN(n_neighbors=k)
    model.fit(data["features"]["int"]["train"], data["y_train"])
    data["features"]["int"]["classifier"]["knn"] = model

    model = kNN(n_neighbors=k)
    model.fit(data["features"]["hog"]["train"], data["y_train"])
    data["features"]["hog"]["classifier"]["knn"] = model

    model = kNN(n_neighbors=k)
    model.fit(data["features"]["cnn"]["train"], data["y_train"])
    data["features"]["cnn"]["classifier"]["knn"] = model

    # Check if the k-NN classifier is stored and can be used in later parts of the assignment
    if not "knn" in data["features"]["int"]["classifier"]:
        print("store the trained knn in data['features']['int']['classifier']['knn'], exiting..")
        sys.exit(-1)
    if not "knn" in data["features"]["hog"]["classifier"]:
        print("store the trained knn in data['features']['hog']['classifier']['knn'], exiting..")
        sys.exit(-1)
    if not "knn" in data["features"]["cnn"]["classifier"]:
        print("store the trained knn in data['features']['cnn']['classifier']['knn'], exiting..")
        sys.exit(-1)

    ## Exercise 2.3: Plot the confusion matrices for all feature and classifier combinations (6 in total)
    # You will need to complete the code below
    if plot:
        _, axes = plt.subplots(nrows=3, ncols=2, figsize=(10,10))
        for i, (feat, items) in enumerate(data["features"].items()):
            for j, (classifier, model) in enumerate(items["classifier"].items()):
                index = j + i * len(items["classifier"].keys())
                ax = axes.flatten()[index]
                ax.axes.get_xaxis().get_label().set_visible(False)
                ax.axes.get_yaxis().get_label().set_visible(False)
                plot_confusion_matrix(model, items["test"], data["y_test"], cmap="Blues", ax=ax)     
                ax.set_title(feat+"_"+classifier)    

        plt.suptitle("Confusion matrices for the feature/classifier combinations")
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.show()
       
    ## Exercise 2.4: Construct a new test set by adding an intensity of 30 to the original test set
    # Recalculate and plot the confusion matrices

    X_test_int_intensity = [increase_intensity(image,30) for image in data["features"]["int"]["test"]]
    X_test_hog_intensity = calculate_hog(X_test_int_intensity)
    model = MobileNet(weights="imagenet", include_top=False)
    X_test_cnn_intensity = calculate_cnn(model, X_test_int_intensity)
    data["features"]["int"]["new_test"] = X_test_int_intensity
    data["features"]["hog"]["new_test"] = X_test_hog_intensity
    data["features"]["cnn"]["new_test"] = X_test_cnn_intensity
    if plot:
        _, axes = plt.subplots(nrows=3, ncols=2, figsize=(10,10))
        for i, (feat, items) in enumerate(data["features"].items()):
            for j, (classifier, model) in enumerate(items["classifier"].items()):
                
                index = j + i * len(items["classifier"].keys())
                ax = axes.flatten()[index]
                ax.axes.get_xaxis().get_label().set_visible(False)
                ax.axes.get_yaxis().get_label().set_visible(False)
                plot_confusion_matrix(model, items["new_test"], data["y_test"], cmap="Blues", ax=ax)     
                ax.set_title(feat+"_"+classifier)    

        plt.suptitle("Confusion matrices for the feature/classifier combinations - Increased Intensity")
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.show()
        
    ## Exercise 2.5: Apply PCA to reduce the dimensionality to 20
    # Use sklearn.decomposition.PCA
    # Recompute and plot the confusion matrices for all feature and classifier combinations (6 in total)
    # Take a look at the code of the previous exercises and use the relevant parts to complete this exercise
    if plot:
        data["features_pca"]={}
        pca = PCA(n_components=20,random_state=42)
        data["features_pca"]["int"] = {}
        data["features_pca"]["hog"] = {}
        data["features_pca"]["cnn"] = {}

        data["features_pca"]["int"]["train"]= pca.fit_transform(data["features"]["int"]["train"])
        data["features_pca"]["int"]["test"] = pca.fit_transform(data["features"]["int"]["test"])

        data["features_pca"]["hog"]["train"] = pca.fit_transform(data["features"]["hog"]["train"])
        data["features_pca"]["hog"]["test"] = pca.fit_transform(data["features"]["hog"]["test"])

        data["features_pca"]["cnn"]["train"] = pca.fit_transform(data["features"]["cnn"]["train"])
        data["features_pca"]["cnn"]["test"] = pca.fit_transform(data["features"]["cnn"]["test"])

        data["features_pca"]["int"]["classifier"] = {}
        data["features_pca"]["hog"]["classifier"] = {}
        data["features_pca"]["cnn"]["classifier"] = {}

        kernel="rbf"
        C=2
        model = SVC(kernel=kernel, C=C)
        model.fit(data["features_pca"]["int"]["train"], data["y_train"])
        data["features_pca"]["int"]["classifier"]["svm"] = model

        model = SVC(kernel=kernel, C=C)
        model.fit(data["features_pca"]["hog"]["train"], data["y_train"])
        data["features_pca"]["hog"]["classifier"]["svm"] = model

        model = SVC(kernel=kernel, C=C)
        model.fit(data["features_pca"]["cnn"]["train"], data["y_train"])
        data["features_pca"]["cnn"]["classifier"]["svm"] = model

        k=1
        model = kNN(n_neighbors=k)
        model.fit(data["features_pca"]["int"]["train"], data["y_train"])
        data["features_pca"]["int"]["classifier"]["knn"] = model

        model = kNN(n_neighbors=k)
        model.fit(data["features_pca"]["hog"]["train"], data["y_train"])
        data["features_pca"]["hog"]["classifier"]["knn"] = model

        model = kNN(n_neighbors=k)
        model.fit(data["features_pca"]["cnn"]["train"], data["y_train"])
        data["features_pca"]["cnn"]["classifier"]["knn"] = model

        _, axes = plt.subplots(nrows=3, ncols=2, figsize=(10,10))
        for i, (feat, items) in enumerate(data["features_pca"].items()):
            for j, (classifier, model) in enumerate(items["classifier"].items()):
                index = j + i * len(items["classifier"].keys())
                ax = axes.flatten()[index]
                ax.axes.get_xaxis().get_label().set_visible(False)
                ax.axes.get_yaxis().get_label().set_visible(False)
                plot_confusion_matrix(model, items["test"], data["y_test"], cmap="Blues", ax=ax)     
                ax.set_title(feat+"_"+classifier)

        plt.suptitle("Confusion matrices for the feature/classifier combinations - PCA")
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.show()
       
    ## Exercise 2.6: Evaluate the accuracy_score for varying values of k of the k-NN
    # Plot the accuracy_score against the k parameter
    if plot:
        ks = [1,5,10,15,20,25,30,35,40,45]
        accuracy = []
        for k in ks:
            model = kNN(n_neighbors=k)
            model.fit(data["features_pca"]["hog"]["train"], data["y_train"])
            y_pred = model.predict(data["features_pca"]["hog"]["test"])
            accuracy.append(accuracy_score(data["y_test"], y_pred))
        plt.plot(ks,accuracy)
        ax=plt.gca()
        ax.set_ylim([0.9,0.95])
        ax.set_xticks(ks)
        plt.show()
       
    ## Exercise 2.7: Evaluate the accuracy_score for varying values of C of the SVM
    # Plot the accuracy_score against the k parameter
    if plot:
        cs = [0.001, 0.005, 0.075, 0.1, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
        accuracy = []
        for k in cs:
            model = SVC(kernel = "rbf", C=k)
            model.fit(data["features"]["hog"]["train"], data["y_train"])
            y_pred = model.predict(data["features"]["hog"]["test"])
            accuracy.append(accuracy_score(data["y_test"], y_pred))
        plt.plot(cs,accuracy)
        plt.show()

    ## Exercise 2.8: Plot ROC curves
    # Create a single plot with the three ROC curves (one for each feature type) with the SVM
    if plot:
        _,ax= plt.subplots(figsize=(10,10))
        model = SVC(kernel = "rbf", C=2)
        model.fit(data["features"]["int"]["train"], data["y_train"])
        plot_roc_curve(model, data["features"]["int"]["test"], data["y_test"], ax=ax, name= "int")
        
        model = SVC(kernel = "rbf", C=2)
        model.fit(data["features"]["hog"]["train"], data["y_train"])
        plot_roc_curve(model, data["features"]["hog"]["test"], data["y_test"], ax=ax, name= "hog")
        
        model = SVC(kernel = "rbf", C=2)
        model.fit(data["features"]["cnn"]["train"], data["y_train"])
        plot_roc_curve(model, data["features"]["cnn"]["test"], data["y_test"], ax=ax, name= "cnn")
        plt.show()
        
    # Save the results to disk to use in later exercises
    np.save("assignment_classification.npy", data)
    
    return data

if __name__ == "__main__":
    assignment_classification(plot=True)
