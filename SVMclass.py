from logitclass import loadData, processData, trainForest
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def trainSVM(X_train, X_test, y_train, y_test, krnl):
    svm = SVC(kernel = krnl)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("Support Vector Machine with", krnl, "kernel")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(conf_matrix)
    print(classification_report(y_test, y_pred))


X, y = loadData()
X_train, X_test, y_train, y_test = processData(X, y)

trainSVM(X_train, X_test, y_train, y_test, 'linear')
trainSVM(X_train, X_test, y_train, y_test, 'rbf')
trainSVM(X_train, X_test, y_train, y_test, 'sigmoid')
trainSVM(X_train, X_test, y_train, y_test, 'poly')
trainForest(X_train, X_test, y_train, y_test)


