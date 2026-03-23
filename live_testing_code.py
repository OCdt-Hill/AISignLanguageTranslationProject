import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model

# ---------------- LOAD MODEL ----------------
model = load_model("gesture_model.keras", compile=False)

actions = ["thank_you", "hello", "bye"]

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---------------- BUFFER ----------------
sequence = []

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #frame = cv2.flip(frame, 1)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    frame_data = np.zeros(42)

    if result.multi_hand_landmarks:
        hand_lms = result.multi_hand_landmarks[0]

        wrist_x = hand_lms.landmark[0].x
        wrist_y = hand_lms.landmark[0].y

        data = []
        for lm in hand_lms.landmark:
            data.append(lm.x - wrist_x)
            data.append(lm.y - wrist_y)

        frame_data = np.array(data)

        mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

    # add frame to sequence
    sequence.append(frame_data)

    # keep last 30 frames
    if len(sequence) > 30:
        sequence = sequence[-30:]

    # ---------------- PREDICTION ----------------
    if len(sequence) == 30:
        input_data = np.expand_dims(sequence, axis=0)

        prediction = model.predict(input_data, verbose=0)[0]
        action = actions[np.argmax(prediction)]

        cv2.putText(frame, action, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()