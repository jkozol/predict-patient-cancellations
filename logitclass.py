import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc#, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, KFold
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
    logit = LogisticRegression(class_weight='balanced', solver='newton-cg')
    logit.fit(X_train, y_train)
    y_pred = logit.predict(X_test)
    return logit
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
    return rf
    """
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("Random Forest")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(conf_matrix)
    print(classification_report(y_test, y_pred))
    """
# Also ROC stuff
def kFoldValidation(X, y, func, k):
    model = None
    max_auc = 0.0
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    kfold = KFold(n_splits=k)
    kfold.get_n_splits(X)
    i = 0
    for train_index, test_index in kfold.split(X):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        mod = func(X_train, X_test, y_train, y_test)
        y_pred = mod.predict(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        AUC = auc(fpr, tpr)
        aucs.append(AUC)
        if AUC > max_auc:
            model = mod
            max_auc = AUC
        y_pred = mod.predict(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, AUC))
        i += 1
    # mean ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    # +/- 1 std ROC
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    # chance line
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc=(1.0, 0.0))
    plt.show()
    return model

X, y = loadData()
print("Logistic Regression")
logit = kFoldValidation(X, y, trainLogit, 10)
print("Random Forest")
rf = kFoldValidation(X, y, trainForest, 10)
