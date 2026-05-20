import cv2
import numpy as np
import mediapipe as mp
from google import genai
from tensorflow.keras.models import load_model
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from dotenv import load_dotenv
import os
import time

load_dotenv()

model = load_model("action_model.h5")
actions = ["YOU", "ME", "WANT","NEED","KNOW","WHAT"]

sentence = []
last_word = ""
final_sentence = ""
predictions=[]

base_options = python.BaseOptions(model_asset_path='/Users/harguneet/Downloads/hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
print("API KEY:", os.getenv("GEMINI_API_KEY"))

def improve_sentence(sentence_list):
    text = " ".join(sentence_list)

    prompt = f"""
    Rewrite this into a grammatically correct sentence.
    Do not summarize. Do not change meaning.

    Input: {text}
    Output:
    """

    for _ in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=[{
                    "role": "user",
                    "parts": [{"text": prompt}]
                }]
            )

            return response.text.strip()

        except Exception as e:
            print("Retrying Gemini...", e)
            time.sleep(2)

    return "Gemini unavailable"

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

cap = cv2.VideoCapture(1)

sequence = []
timestamp = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    timestamp +=1 
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect_for_video(mp_image, timestamp)

    if result.hand_landmarks:
        keypoints = extract_keypoints(result)
        sequence.append(keypoints)
    else:
        sequence = []  # reset if no hands
    sequence = sequence[-30:]

    if len(sequence) == 30:
        arr = np.array(sequence)
        print(arr.ndim)
        res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
        print(res.ndim)
        if np.max(res) > 0.8:
            word = actions[np.argmax(res)]
            predictions.append(np.argmax(res))

        if len(predictions) > 10:
            predictions = predictions[-10:]

        if len(predictions) > 0 and predictions.count(predictions[-1]) > 7:
            word = actions[predictions[-1]]
        
            if word != last_word:
                sentence.append(word)
                last_word = word

    cv2.putText(frame, " ".join(sentence), (50, 200),
        cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0), 3)
    
    cv2.putText(frame, final_sentence, (50, 250),
            cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 3)
    cv2.imshow("Prediction", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        if len(sentence) > 0:
            try:
                final_sentence = improve_sentence(sentence)
            except Exception as e:
                print("Gemini error:", e)
                final_sentence = "..."
            print("Gemini:", final_sentence)

            sentence = []
            last_word = ""

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
