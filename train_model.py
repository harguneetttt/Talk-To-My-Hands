import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

DATA_PATH = "data"
actions = ["YOU", "ME", "WANT","NEED","KNOW","WHAT"]

X = []
y = []

for action_idx, action in enumerate(actions):
    folder = os.path.join(DATA_PATH, action)

    for file in os.listdir(folder):
        if file.endswith(".npy"):
            sequence = np.load(os.path.join(folder, file))
            X.append(sequence)
            y.append(action_idx)

X = np.array(X)
y = to_categorical(y)

print("Training on:", X.shape)

model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(30,126)))
model.add(LSTM(128, return_sequences=False))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X, y, epochs=50, batch_size=8)

model.save("action_model.h5")

print("Model trained and saved.")
