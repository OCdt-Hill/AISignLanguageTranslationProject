import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# ---------------- CONFIG ----------------
DATA_PATH = "MP_Data"
actions = ["thank_you", "hello", "bye"]

SEQUENCE_LENGTH = 30
FEATURES = 42

# ---------------- LABEL MAP ----------------
label_map = {label: num for num, label in enumerate(actions)}

# ---------------- LOAD DATA ----------------
sequences, labels = [], []

for action in actions:
    action_path = os.path.join(DATA_PATH, action)

    if not os.path.exists(action_path):
        continue

    for sequence in os.listdir(action_path):
        window = []

        for frame_num in range(SEQUENCE_LENGTH):
            file_path = os.path.join(action_path, sequence, f"{frame_num}.npy")

            if os.path.exists(file_path):
                res = np.load(file_path)
            else:
                res = np.zeros(FEATURES)

            window.append(res)

        sequences.append(window)
        labels.append(label_map[action])

# ---------------- CONVERT TO NUMPY ----------------
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# ---------------- TRAIN / TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL ----------------
model = Sequential()

model.add(Input(shape=(SEQUENCE_LENGTH, FEATURES)))

model.add(LSTM(64, return_sequences=True, activation='relu'))
model.add(Dropout(0.2))

model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))

# ---------------- COMPILE ----------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ---------------- TRAIN ----------------
model.fit(
    X_train,
    y_train,
    epochs=30,
    validation_data=(X_test, y_test)
)

# ---------------- SAVE MODEL ----------------
model.save("gesture_model.keras")

print("Model training complete and saved!")