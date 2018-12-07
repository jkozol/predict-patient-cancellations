import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, KFold

def loadData():
    dataset = pd.read_csv('data/data_train.csv')
    #feature_cols = ['Date Diff', 'SMS', 'Email', 'Gender', 'Age']
    #target = 'No Show/LateCancel Flag'
    X = dataset.drop(columns=['Patient Id', 'No Show/LateCancel Flag'])
    y = dataset['No Show/LateCancel Flag']
    return(X, y)


def trainSVM(X_train, X_test, y_train, y_test, krnl):
    svm = SVC(kernel = krnl)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    return (auc, svm)

def SVM_KFoldValidation(X, y, func, k, krnl):
    model = None
    max_auc = 0.0
    aucs = np.array([])
    kfold = KFold(n_splits=k)
    kfold.get_n_splits(X)
    for train_index, test_index in kfold.split(X):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]
        auc, mod = func(X_train, X_test, y_train, y_test, krnl)
        aucs = np.append(aucs, auc)
        if auc > max_auc:
            model = mod
            max_auc = auc

    return (np.mean(aucs), model)

X,y = loadData()
auc_svml, svml = kFoldValidation(X, y, trainSVM, 10, 'linear')
print("Linear SVM AUC: ", auc_svml)
auc_svmr, svmr = kFoldValidation(X, y, trainSVM, 10, 'rbf')
print("RBF SVM AUC: ", auc_svmr)
auc_svms, svms = kFoldValidation(X, y, trainSVM, 10, 'sigmoid')
print("Sigmoid SVM AUC: ", auc_svms)
auc_svmp, svmp = SVM_KFoldValidation(X, y, trainSVM, 10, 'poly')
print("Polynomial SVM AUC: ", auc_svmp)
