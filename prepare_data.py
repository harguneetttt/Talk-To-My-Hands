import numpy as np
import os
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

print(X)
print("X shape:", X.shape)
print("y shape:", y.shape)
