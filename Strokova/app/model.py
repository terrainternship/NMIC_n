from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def create_model():
    model = Sequential()
    model.add(Dense(32, activation='relu'))
    model.add(Dense(800, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def training(x_train, x_test, y_train, y_test):
    model = create_model()
    history = model.fit(x_train, y_train, epochs=8, batch_size=64, validation_data=(x_test, y_test))
    model_weights_path = 'model_weights.h5'
    model.save_weights(model_weights_path)
    return model_weights_path

