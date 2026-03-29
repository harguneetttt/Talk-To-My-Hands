import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

X = np.load("data/X_seq.npy")
y = np.load("data/y_seq.npy")

model = Sequential()

model.add(LSTM(64, return_sequences=True, input_shape=(30,195)))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64))

model.add(Dense(64, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X, y, epochs=30, batch_size=16)

model.save("model.h5")

print("Model trained and saved")