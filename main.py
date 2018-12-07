import numpy as np
from logitClassifier import trainLogit
from rfClassifier import trainForest
from svmClassifier import linearSVM, rbfSVM, sigmoidSVM
from nnClassifier import neuralnet
from utils import loadData, kFoldValidation

X, y = loadData()

classifiers = [trainLogit, trainForest, linearSVM, rbfSVM, sigmoidSVM, neuralnet]
models = [kFoldValidation(X, y, model, 10) for model in classifiers]
print(models)
# print("Logistic Regression")
# logit, logitAUC = kFoldValidation(X, y, trainLogit, 10)
# print("Random Forest")
# rf, rfAUC = kFoldValidation(X, y, trainForest, 10)
