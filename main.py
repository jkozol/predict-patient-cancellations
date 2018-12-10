import numpy as np
import pandas as pd
from logitClassifier import logit
from rfClassifier import randForest
from svmClassifier import linearSVM, rbfSVM, sigmoidSVM
from nnClassifier import neuralnet
from utils import loadTrainingData, loadTestData, kFoldValidation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def selectModel():
    X, y = loadTrainingData()
    classifiers = [logit, randForest, linearSVM, rbfSVM, sigmoidSVM, neuralnet]
    models = [kFoldValidation(X, y, model, 10) for model in classifiers]
    for (model, auc) in models:
        y_pred = np.rint(model.predict(X).flatten())
        print("Accuracy:", accuracy_score(y, y_pred))
        print("AUC:", auc)
        print(classification_report(y, y_pred))

    best_model = max(models, key=lambda x:x[1])[0]
    return best_model

def predict(model):
    X, patientIds = loadTestData()
    predictions = model.predict(X).flatten()
    predictions_df = pd.DataFrame({'Patient Id': patientIds, 'Expected': predictions}).drop_duplicates('Patient Id')

    return predictions_df

def run():
    model = selectModel()
    predictions = np.rint(predict(model))
    predictions.to_csv('data/predictions.csv', index=False)

run()
