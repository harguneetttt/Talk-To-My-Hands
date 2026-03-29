import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

fingers = [
[5,6,8],
[9,10,12],
[13,14,16],
[17,18,20]
]
thumb_tip = 4

HAND_CONNECTIONS = [
(0,1),(1,2),(2,3),(3,4),
(0,5),(5,6),(6,7),(7,8),
(5,9),(9,10),(10,11),(11,12),
(9,13),(13,14),(14,15),(15,16),
(13,17),(17,18),(18,19),(19,20),
(0,17)
]

base_options = python.BaseOptions(model_asset_path='/Users/harguneet/Downloads/hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options = base_options, num_hands=2, running_mode = vision.RunningMode.VIDEO)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(1)

timestamp = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    timestamp+=1

    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = new_frame)
    hand_landmarker_result = detector.detect_for_video(mp_image, timestamp)

    if hand_landmarker_result.hand_landmarks:
        for hand_landmarks in hand_landmarker_result.hand_landmarks:
            finger_fold_status=[]

            h, w, c = frame.shape

            for lm in hand_landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)

                cv2.circle(frame, (x, y), 12, (255, 213, 63), cv2.FILLED)
            
            angles=[]
            for base,joint,tip in fingers:
                A = (hand_landmarks[base].x, hand_landmarks[base].y)
                B = (hand_landmarks[joint].x, hand_landmarks[joint].y)
                C = (hand_landmarks[tip].x, hand_landmarks[tip].y)
                A = np.array(A)
                B = np.array(B)
                C = np.array(C)

                BA = A - B
                lenBA = np.linalg.norm(BA)
                BC = C - B
                lenBC = np.linalg.norm(BC)

                cos_angle = np.dot(BA,BC) / (lenBA * lenBC)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                final = np.degrees(angle)
                
                angles.append(final)
            # print(angles)
            finger_state=[]
            for i in angles:
                if(i>150):
                    finger_state.append(1)
                else:
                    finger_state.append(0)
            # print(finger_state)

            if hand_landmarks[thumb_tip].y < hand_landmarks[thumb_tip-1].y < hand_landmarks[thumb_tip-2].y:
                if finger_state == [0,0,0,0]:
                    cv2.putText(frame, "LIKE", (100, 120), cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 0, 0), 3)

            if hand_landmarks[thumb_tip].y > hand_landmarks[thumb_tip-1].y > hand_landmarks[thumb_tip-2].y:
                if finger_state == [0,0,0,0]:
                    cv2.putText(frame, "DISLIKE", (100, 120), cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 0, 0), 3)


            for start_idx, end_idx in HAND_CONNECTIONS:
                x1 = int(hand_landmarks[start_idx].x * w)
                y1 = int(hand_landmarks[start_idx].y * h)

                x2 = int(hand_landmarks[end_idx].x * w)
                y2 = int(hand_landmarks[end_idx].y * h)

                cv2.line(frame, (x1, y1), (x2, y2), (255,0,0), 4)

    cv2.imshow("Output", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
