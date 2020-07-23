from models_partc import logistic_regression_pred
from sklearn.model_selection import KFold, ShuffleSplit
from numpy import mean
from sklearn.metrics import *
import utils

# USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

# USE THIS RANDOM STATE FOR ALL OF YOUR CROSS VALIDATION TESTS, OR THE TESTS WILL NEVER PASS
RANDOM_STATE = 545510477

#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_kfold(X,Y,k=5):
    #TODO:First get the train indices and test indices for each iteration
    #Then train the classifier accordingly
    #Report the mean accuracy and mean auc of all the folds
    cv = KFold(n_splits=k, random_state=RANDOM_STATE)
    accuracy = 0
    AUC = 0
    for train, test in cv.split(X, Y):
        y_pred = logistic_regression_pred(X[train], Y[train], X[test])
        accuracy += accuracy_score(Y[test],y_pred)
        fpr, tpr, thresholds = roc_curve(Y[test], y_pred)
        AUC += auc(fpr, tpr)
    accuracy = accuracy/k
    AUC = AUC/k
    
    return accuracy,AUC


#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_randomisedCV(X,Y,iterNo=5,test_percent=0.2):
    #TODO: First get the train indices and test indices for each iteration
    #Then train the classifier accordingly
    #Report the mean accuracy and mean auc of all the iterations
    cv = ShuffleSplit(n_splits=iterNo, test_size = test_percent, random_state=RANDOM_STATE)
    accuracy = 0
    AUC = 0
    for train, test in cv.split(X, Y):
        y_pred = logistic_regression_pred(X[train], Y[train], X[test])
        accuracy += accuracy_score(Y[test],y_pred)
        fpr, tpr, thresholds = roc_curve(Y[test], y_pred)
        AUC += auc(fpr, tpr)
    accuracy = accuracy/iterNo
    AUC = AUC/iterNo
       
    return accuracy,AUC


def main():
	X,Y = utils.get_data_from_svmlight("C:/users/yyan/Downloads/homework1/deliverables/features_svmlight.train")
	print("Classifier: Logistic Regression__________")
	acc_k,auc_k = get_acc_auc_kfold(X,Y)
	print(("Average Accuracy in KFold CV: "+str(acc_k)))
	print(("Average AUC in KFold CV: "+str(auc_k)))
	acc_r,auc_r = get_acc_auc_randomisedCV(X,Y)
	print(("Average Accuracy in Randomised CV: "+str(acc_r)))
	print(("Average AUC in Randomised CV: "+str(auc_r)))

if __name__ == "__main__":
	main()

