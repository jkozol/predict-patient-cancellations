from sklearn.ensemble import RandomForestClassifier

def randForest(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf.fit(X_train, y_train)
    return rf
