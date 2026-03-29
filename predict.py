import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------------------------------
# LOAD MODEL
# -------------------------------
model = load_model("model.h5")

actions = list(np.load("data/y.npy", allow_pickle=True))
actions = sorted(list(set(actions)))
last_prediction_time = 0
display_action = ""
display_duration = 2000  # milliseconds (2 seconds)
# -------------------------------
# LOAD MEDIAPIPE MODELS
# -------------------------------
pose_base = python.BaseOptions(
    model_asset_path="models/pose_landmarker_full.task"
)

pose_options = vision.PoseLandmarkerOptions(
    base_options=pose_base,
    running_mode=vision.RunningMode.VIDEO
)

pose_detector = vision.PoseLandmarker.create_from_options(pose_options)


hand_base = python.BaseOptions(
    model_asset_path="models/hand_landmarker.task"
)

hand_options = vision.HandLandmarkerOptions(
    base_options=hand_base,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO
)

hand_detector = vision.HandLandmarker.create_from_options(hand_options)

# -------------------------------
# LANDMARKS
# -------------------------------
POSE_POINTS = [11,12,13,14,15,16,23,24,25,26,27,28]
FACE_POINTS = [0,1,2,3,4,5,6,7,8,9,10]

EXPECTED = 195

sequence = []
threshold = 0.8
timestamp = 0

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    timestamp += 1

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    coords = []

    # -------------------------------
    # POSE + FACE
    # -------------------------------
    pose_result = pose_detector.detect_for_video(mp_image, timestamp)

    if pose_result.pose_landmarks:
        landmarks = pose_result.pose_landmarks[0]

        for idx in POSE_POINTS:
            lm = landmarks[idx]
            coords.extend([lm.x, lm.y, lm.z])

        for idx in FACE_POINTS:
            lm = landmarks[idx]
            coords.extend([lm.x, lm.y, lm.z])
    else:
        coords.extend([0]*((len(POSE_POINTS)+len(FACE_POINTS))*3))

    # -------------------------------
    # HANDS
    # -------------------------------
    hand_result = hand_detector.detect_for_video(mp_image, timestamp)

    if hand_result.hand_landmarks:
        for hand in hand_result.hand_landmarks:
            for lm in hand:
                coords.extend([lm.x, lm.y, lm.z])

        if len(hand_result.hand_landmarks) == 1:
            coords.extend([0]*63)
    else:
        coords.extend([0]*126)

    # -------------------------------
    # FIX SIZE
    # -------------------------------
    coords = coords[:EXPECTED]

    if len(coords) < EXPECTED:
        coords.extend([0]*(EXPECTED - len(coords)))

    # -------------------------------
    # SEQUENCE
    # -------------------------------
    sequence.append(coords)
    sequence = sequence[-30:]

    current_time = cv2.getTickCount() / cv2.getTickFrequency() * 1000  # ms

    if len(sequence) == 30 and (current_time - last_prediction_time > 2000):

        res = model.predict(np.expand_dims(sequence, axis=0))[0]

        if res.max() > threshold:
            display_action = actions[np.argmax(res)]
            last_prediction_time = current_time
    if display_action != "":
        cv2.putText(frame, display_action, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Prediction", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()