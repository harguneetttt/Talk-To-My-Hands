import numpy as np
from tensorflow.keras.utils import to_categorical

X = np.load("data/X.npy")
y = np.load("data/y.npy")

sequence_length = 30

sequences = []
labels = []

label_map = {label: idx for idx, label in enumerate(np.unique(y))}

for i in range(len(X) - sequence_length):
    seq = X[i:i+sequence_length]
    label = y[i]

    sequences.append(seq)
    labels.append(label_map[label])

X_seq = np.array(sequences)
y_seq = to_categorical(labels)

np.save("data/X_seq.npy", X_seq)
np.save("data/y_seq.npy", y_seq)

print("Sequence shape:", X_seq.shape)   # (?, 30, 195)
print("Labels shape:", y_seq.shape)