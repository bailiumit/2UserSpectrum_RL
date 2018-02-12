import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Generate dummy data
import numpy as np

model = Sequential()
model.add(Dense(30, activation='relu', input_dim=20))
model.add(Dense(2, activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=PG,
      optimizer=sgd]) 

for i in range(M)
    for j in range(N)
        sample = Generate()
        actionset = predict(sample)
        x = x.append(sample)
        y = -CalY(actionset, x)

    model.fit(x, y,
            epochs=1,
            batch_size=N*T) # size of x




def PG:






# x_train = np.random.random((1000, 20))
# y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
# x_test = np.random.random((100, 20))
# y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

# model = Sequential()
# # Dense(64) is a fully-connected layer with 64 hidden units.
# # in the first layer, you must specify the expected input data shape:
# # here, 20-dimensional vectors.
# model.add(Dense(30, activation='relu', input_dim=20))
# model.add(Dense(2, activation='softmax'))

# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy',
#               optimizer=sgd,
#               metrics=['accuracy'])

# model.fit(x_train, y_train,
#           epochs=1,
#           batch_size=) # size of x
# score = model.evaluate(x_test, y_test, batch_size=128)