import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

VIDEO_ROOT = "source"
DATA_ROOT = "data"
actions = ["YOU", "ME", "WANT","NEED","KNOW","WHAT"]

base_options = python.BaseOptions(model_asset_path='model')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)
timestamp = 0

def extract_keypoints(result):
    if result.hand_landmarks:
        hands = []
        for hand in result.hand_landmarks:
            for lm in hand:
                hands.extend([lm.x, lm.y, lm.z])
        while len(hands) < 126:
            hands.extend([0,0,0])
        return np.array(hands)
    else:
        return np.zeros(126)

sequence_length = 30

for action in actions:
    video_folder = os.path.join(VIDEO_ROOT, action)
    output_folder = os.path.join(DATA_ROOT, action)

    os.makedirs(output_folder, exist_ok=True)

    seq_num = len(os.listdir(output_folder))

    for video_name in os.listdir(video_folder):
        video_path = os.path.join(video_folder, video_name)

        cap = cv2.VideoCapture(video_path)
        sequence = []

        print(f"Processing: {video_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp += 1

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = detector.detect_for_video(mp_image, timestamp)

            keypoints = extract_keypoints(result)
            sequence.append(keypoints)

            if len(sequence) == sequence_length:
                np.save(os.path.join(output_folder, f"{seq_num}.npy"), sequence)
                seq_num += 1
                sequence = []

        cap.release()

print("All videos processed.")
