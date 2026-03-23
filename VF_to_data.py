import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ---------------- CONFIG ----------------
DATA_PATH = "MP_Data"
ACTION = "hello"
NO_SEQUENCES = 30      # number of recordings
SEQUENCE_LENGTH = 30   # frames per recording

# ---------------- SETUP ----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Create folders
for sequence in range(NO_SEQUENCES):
    os.makedirs(os.path.join(DATA_PATH, ACTION, str(sequence)), exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)

# ---------------- DATA COLLECTION LOOP ----------------
for sequence in range(NO_SEQUENCES):

    for frame_num in range(SEQUENCE_LENGTH):

        ret, frame = cap.read()
        if not ret:
            break

        # Mirror view
        #frame = cv2.flip(frame, 1)

        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        # Draw landmarks
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        # UI text
        if frame_num == 0:
            cv2.putText(frame, 'STARTING COLLECTION', (120, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.putText(frame, f'Action: {ACTION} | Seq: {sequence}', (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Data Collection', frame)
            cv2.waitKey(2000)
        else:
            cv2.putText(frame, f'Action: {ACTION} | Seq: {sequence}', (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Data Collection', frame)

        # ---------------- FEATURE EXTRACTION ----------------
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

        # ---------------- SAVE DATA ----------------
        save_path = os.path.join(
            DATA_PATH,
            ACTION,
            str(sequence),
            str(frame_num)
        )

        np.save(save_path, frame_data)

        # Exit on ESC
        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            exit()

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()