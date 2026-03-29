import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd

base_options = python.BaseOptions(model_asset_path='/Users/harguneet/Downloads/face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options = base_options,output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       running_mode = vision.RunningMode.VIDEO,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)


dataset=[]
cap = cv2.VideoCapture(1)
timestamp=0
while True:
    ret,frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame,1)
    h, w, c = frame.shape
    new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    timestamp+=1

    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = new_frame)
    face_landmarker_result = detector.detect_for_video(mp_image, timestamp)

    if face_landmarker_result.face_landmarks:
        for face_landmarks in face_landmarker_result.face_landmarks:
            # this is for drawing
            for fifi in face_landmarks:
                x = int(fifi.x * w)
                y = int(fifi.y * h)

                cv2.circle(frame, (x, y), 1, (255, 20, 0), -1)
            store=[]
            # this is for storing
            for fifi in face_landmarks:
                x = fifi.x
                y = fifi.y
                z = fifi.z
                store.append(x)
                store.append(y)
                store.append(z)
                # print(len(store))
            dataset.append(store)
            # print(len(dataset))

    cv2.imshow("Output", frame)

    if cv2.waitKey(1) == ord('q'):
        break
df = pd.DataFrame(dataset)
df.to_csv("face_dataset.csv", index=False)

cap.release()

cv2.destroyAllWindows()
