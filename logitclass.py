import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def loadData():
    dataset = pd.read_csv('data/data.csv')
    feature_cols = ['Date Diff', 'SMS', 'Email', 'Gender']
    target = 'No Show/LateCancel Flag'
    X = dataset[feature_cols]
    y = dataset[target]
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

    conf_matrix = confusion_matrix(y_test, y_pred)

    print("Logistic Regression")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(conf_matrix)
    print(classification_report(y_test, y_pred))

def trainForest(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    conf_matrix = confusion_matrix(y_test, y_pred)

    print("Random Forest")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(conf_matrix)
    print(classification_report(y_test, y_pred))

X, y = loadData()
X_train, X_test, y_train, y_test = processData(X, y)
trainLogit(X_train, X_test, y_train, y_test)
trainForest(X_train, X_test, y_train, y_test)
