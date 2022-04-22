import pickle
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten

pickle_in = open("X_micro_expressions.pickle", "rb")
X = pickle.load(pickle_in)
X = X / 255.0

pickle_in = open("y_micro_expressions.pickle", "rb")
y = pickle.load(pickle_in)

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(1024))
model.add(Activation("relu"))

model.add(Dense(512))
model.add(Activation("relu"))

model.add(Dense(256))
model.add(Activation("relu"))

model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(10))
model.add(Activation('softmax'))
model.add(Dropout(0.2))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X, y, batch_size=32, epochs=15, validation_split=0.3)

model.save('32x2-model.model')
