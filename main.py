import numpy as np
import pandas as pd
from logitClassifier import trainLogit
from rfClassifier import trainForest
from svmClassifier import linearSVM, rbfSVM, sigmoidSVM
from nnClassifier import neuralnet
from utils import loadTrainingData, loadTestData, kFoldValidation

def selectModel():
    X, y = loadTrainingData()
    classifiers = [trainLogit, trainForest, linearSVM, rbfSVM, sigmoidSVM, neuralnet]
    models = [kFoldValidation(X, y, model, 10) for model in classifiers]
    best_model = max(models, key=lambda x:x[1])[0]
    return best_model

def predict(model):
    X, patientIds = loadTestData()
    predictions = model.predict(X)
    predictions_df = pd.DataFrame({'Patient Id': patientIds, 'No Show/LateCancel Flag': predictions})
    return predictions_df

def run():
    model = selectModel()
    predictions = predict(model)
    predictions.to_csv('data/predictions.csv')

run()
