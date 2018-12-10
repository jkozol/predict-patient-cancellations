from keras.models import Sequential
from keras.layers import Dense

def neuralnet(X_train, X_test, y_train, y_test):

    classifier = Sequential()

    classifier.add(Dense(activation='relu', input_dim=336, units=167, kernel_initializer='uniform'))
    classifier.add(Dense(activation='relu', units=167, kernel_initializer='uniform'))
    classifier.add(Dense(activation='sigmoid', units=1, kernel_initializer='uniform'))

    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    classifier.fit(X_train, y_train, batch_size=300, epochs=200)

    return classifier
