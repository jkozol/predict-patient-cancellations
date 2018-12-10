from sklearn.svm import SVC

def linearSVM(X_train, X_test, y_train, y_test):
    svm = SVC(kernel = 'linear')
    svm.fit(X_train, y_train)
    return svm

def rbfSVM(X_train, X_test, y_train, y_test):
    svm = SVC(kernel = 'rbf')
    svm.fit(X_train, y_train)
    return svm

def sigmoidSVM(X_train, X_test, y_train, y_test):
    svm = SVC(kernel = 'sigmoid')
    svm.fit(X_train, y_train)
    return svm

def polynomialSVM(X_train, X_test, y_train, y_test):
    svm = SVC(kernel = 'poly')
    svm.fit(X_train, y_train)
    return svm
