import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report, confusion_matrix, r2_score

def trainLogit(X_train, X_test, y_train, y_test):
    logit = LogisticRegression(class_weight='balanced', solver='newton-cg')
    logit.fit(X_train, y_train)
    # y_pred = logit.predict(X_test)
    # # """
    # conf_matrix = confusion_matrix(y_test, y_pred)
    #
    # print("Logistic Regression")
    # print("Accuracy:", accuracy_score(y_test, y_pred))
    # print(conf_matrix)
    # print(classification_report(y_test, y_pred))
    # # """
    return logit
