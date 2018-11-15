import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def loadData():
    dataset = pd.read_csv('data/data.csv')
    feature_cols = ['Date Diff', 'SMS', 'Email', 'Gender']
    target = 'No Show/LateCancel Flag'
    X = dataset[feature_cols]
    y = dataset[target]
    return(X, y)

def trainLogit(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    logit = LogisticRegression(class_weight='balanced')
    model = logit.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    conf_matrix = confusion_matrix(y_test, y_pred)

    print("Logistic Regression")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(conf_matrix)
    print(classification_report(y_test, y_pred))

def trainForest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


    forest = RandomForestClassifier(n_estimators=100)
    model = forest.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    conf_matrix = confusion_matrix(y_test, y_pred)

    print("Random Forest")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(conf_matrix)
    print(classification_report(y_test, y_pred))

X, y = loadData()
trainLogit(X, y)
trainForest(X, y)
