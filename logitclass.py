import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
#import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, #accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier#, RandomForestRegressor
#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA

def loadData():
    dataset = pd.read_csv('data/data_train.csv')
    #feature_cols = ['Date Diff', 'SMS', 'Email', 'Gender', 'Age']
    #target = 'No Show/LateCancel Flag'
    X = dataset.drop(columns=['Patient Id', 'No Show/LateCancel Flag'])
    y = dataset['No Show/LateCancel Flag']
    return(X, y)

def processData(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)

    X_train_bal1 = X_train[y_train == 1]
    y_train1 = y_train[y_train == 1]
    X_train_bal0 = X_train[y_train == 0].sample(len(X_train_bal1))
    y_train0 = y_train[y_train == 0].sample(len(X_train_bal1))
    print(X_train_bal0.size, X_train_bal1.size)

    X_train = pd.concat([X_train_bal1, X_train_bal0])
    y_train = pd.concat([y_train1, y_train0])
    return(X_train, X_test, y_train, y_test)

def trainLogit(X_train, X_test, y_train, y_test):
    logit = LogisticRegression(class_weight='balanced')
    logit.fit(X_train, y_train)
    y_pred = logit.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    return (auc, logit)
    """
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("Logistic Regression")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(conf_matrix)
    print(classification_report(y_test, y_pred))
    """

def trainForest(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    return (auc, rf)
    """
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("Random Forest")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(conf_matrix)
    print(classification_report(y_test, y_pred))
    """

def kFoldValidation(X, y, k):
    model_logit = None
    model_rf = None
    max_auc_logit = 0.0
    max_auc_rf = 0.0
    logit_aucs = np.array([])
    rf_aucs = np.array([])
    kfold = KFold(X.shape[0], n_folds=k)
    
    for train_index, test_index in kfold:
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]
        
        auc_logit, logit = trainLogit(X_train, X_test, y_train, y_test)
        logit_aucs = np.append(logit_aucs, auc_logit)
        if auc_logit > max_auc_logit:
            model_logit = logit
            max_auc_logit = auc_logit
        
        auc_rf, rf = trainForest(X_train, X_test, y_train, y_test)
        rf_aucs = np.append(rf_aucs, auc_rf)
        if auc_rf > max_auc_rf:
            model_rf = rf
            max_auc_rf = auc_rf
    return (np.mean(logit_aucs), np.mean(rf_aucs), model_logit, model_rf)

X, y = loadData()
auc_logit, auc_rf, logit, rf = kFoldValidation(X, y, 15)
print("Logistic Regression AUC: ", auc_logit)
print("Random Forest AUC: ", auc_rf)
