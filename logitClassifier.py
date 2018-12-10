from sklearn.linear_model import LogisticRegression

def logit(X_train, X_test, y_train, y_test):
    logit = LogisticRegression(class_weight='balanced', solver='newton-cg')
    logit.fit(X_train, y_train)
    return logit
