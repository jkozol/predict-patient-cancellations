import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

from sklearn.linear_model import SGDClassifier

def loadData():
    dataset = pd.read_csv('data/data.csv')
    feature_cols = ['Date Diff']
    target = 'No Show/LateCancel Flag'
    X = dataset[feature_cols]
    y = dataset[target]
    print(X,y)

def train(X, y):
    old_stdout = sys.stdout
    # sys.stdout = mystdout = StringIO()
    clf = SGDClassifier(verbose=1)
    clf.fit(X, y)
    sys.stdout = old_stdout
    loss_history = mystdout.getvalue()
    loss_list = []
    for line in loss_history.split('\n'):
        if(len(line.split("loss: ")) == 1):
            continue
        loss_list.append(float(line.split("loss: ")[-1]))
    pass

loadData()
