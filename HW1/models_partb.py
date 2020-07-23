import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *

import utils

# setup the randoms tate
RANDOM_STATE = 545510477

#input: X_train, Y_train
#output: Y_pred
def logistic_regression_pred(X_train, Y_train):
    lr = LogisticRegression(random_state=RANDOM_STATE)
    lr.fit(X_train, Y_train)
    y_pred = lr.predict(X_train)
    return y_pred

#input: X_train, Y_train
#output: Y_pred
def svm_pred(X_train, Y_train):
    clf = LinearSVC(random_state=RANDOM_STATE)
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_train)
    return y_pred

#input: X_train, Y_train
#output: Y_pred
def decisionTree_pred(X_train, Y_train):
    clf = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth = 5)
    clf.fit(X_train,Y_train)
    y_pred = clf.predict(X_train)
    return y_pred

#input: Y_pred,Y_true
#output: accuracy, auc, precision, recall, f1-score
def classification_metrics(Y_pred, Y_true):
    accuracy = accuracy_score(Y_true, Y_pred)
    fpr, tpr, thresholds = roc_curve(Y_true, Y_pred)
    AUC = auc(fpr, tpr)
    precision = precision_score(Y_true, Y_pred)
    recall = recall_score(Y_true, Y_pred)
    f1score = f1_score(Y_true, Y_pred)
    return accuracy, AUC, precision, recall, f1score

#input: Name of classifier, predicted labels, actual labels
def display_metrics(classifierName,Y_pred,Y_true):
	print("______________________________________________")
	print(("Classifier: "+classifierName))
	acc, auc_, precision, recall, f1score = classification_metrics(Y_pred,Y_true)
	print(("Accuracy: "+str(acc)))
	print(("AUC: "+str(auc_)))
	print(("Precision: "+str(precision)))
	print(("Recall: "+str(recall)))
	print(("F1-score: "+str(f1score)))
	print("______________________________________________")
	print("")

def main():
	X_train, Y_train = utils.get_data_from_svmlight("C:/users/yyan/Downloads/homework1/deliverables/features_svmlight.train")
	
	display_metrics("Logistic Regression",logistic_regression_pred(X_train,Y_train),Y_train)
	display_metrics("SVM",svm_pred(X_train,Y_train),Y_train)
	display_metrics("Decision Tree",decisionTree_pred(X_train,Y_train),Y_train)
	

if __name__ == "__main__":
	main()
	
