import os
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------------------------------
# LOAD MODELS
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
# DATASET PATH
# -------------------------------
DATASET_PATH = r"D:\shisha\data"

# -------------------------------
# LANDMARK SELECTION
# -------------------------------
POSE_POINTS = [11,12,13,14,15,16,23,24,25,26,27,28]
FACE_POINTS = [0,1,2,3,4,5,6,7,8,9,10]

EXPECTED = 195

X = []
y = []

timestamp = 0

# -------------------------------
# LOOP DATASET
# -------------------------------
for gesture in os.listdir(DATASET_PATH):

    gesture_path = os.path.join(DATASET_PATH, gesture)
    if not os.path.isdir(gesture_path):
        continue

    print("Processing:", gesture)

    for video in os.listdir(gesture_path):

        cap = cv2.VideoCapture(os.path.join(gesture_path, video))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp += 1

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            coords = []

            # -------------------------------
            # POSE + FACE (from pose model)
            # -------------------------------
            pose_result = pose_detector.detect_for_video(mp_image, timestamp)

            if pose_result.pose_landmarks:
                landmarks = pose_result.pose_landmarks[0]

                # Pose
                for idx in POSE_POINTS:
                    lm = landmarks[idx]
                    coords.extend([lm.x, lm.y, lm.z])

                # Face
                for idx in FACE_POINTS:
                    lm = landmarks[idx]
                    coords.extend([lm.x, lm.y, lm.z])
            else:
                coords.extend([0] * ( (len(POSE_POINTS)+len(FACE_POINTS)) * 3 ))

            # -------------------------------
            # HANDS (1 or 2 hands)
            # -------------------------------
            hand_result = hand_detector.detect_for_video(mp_image, timestamp)

            if hand_result.hand_landmarks:

                for hand in hand_result.hand_landmarks:
                    for lm in hand:
                        coords.extend([lm.x, lm.y, lm.z])

                # if only 1 hand → pad second
                if len(hand_result.hand_landmarks) == 1:
                    coords.extend([0]*63)

            else:
                coords.extend([0]*126)

            # -------------------------------
            # FINAL SIZE FIX
            # -------------------------------
            coords = coords[:EXPECTED]

            if len(coords) < EXPECTED:
                coords.extend([0]*(EXPECTED - len(coords)))

            X.append(coords)
            y.append(gesture)

        cap.release()

# -------------------------------
# SAVE
# -------------------------------
X = np.array(X)
y = np.array(y)

os.makedirs("data", exist_ok=True)

np.save("data/X.npy", X)
np.save("data/y.npy", y)

print("Saved:", X.shape, y.shape)