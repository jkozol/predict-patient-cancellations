import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from keras.models import Sequential
from keras.layers import Dense

def loadData():
    dataset = pd.read_csv('data/data_train.csv')
    X = dataset.drop(columns=['Patient Id', 'No Show/LateCancel Flag'])
    y = dataset['No Show/LateCancel Flag']
    return X, y


def processData(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return(X_train, X_test, y_train, y_test)


def neuralnet(X_train, X_test, y_train, y_test):

    classifier = Sequential()

    classifier.add(Dense(activation='relu', input_dim=334, units=167, kernel_initializer='uniform'))
    classifier.add(Dense(activation='relu', units=167, kernel_initializer='uniform'))
    classifier.add(Dense(activation='sigmoid', units=1, kernel_initializer='uniform'))

    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    classifier.fit(X_train, y_train, batch_size=300, epochs=200)

    return classifier

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


# X, y = loadData()
# X_train, X_test, y_train, y_test = processData(X, y)
# validate = kFoldValidation(X, y, nnclassifier, 10)
