import base64
import asyncio
import json
import cv2
import numpy as np
import mediapipe as mp
import websockets
import time
from tensorflow.keras.models import load_model
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from dotenv import load_dotenv
import os
from google import genai
load_dotenv()
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)
MODEL_PATH = "action_model.h5"
HAND_MODEL_PATH = "models/hand_landmarker.task"
ACTIONS = ["YOU", "ME", "WANT", "NEED", "KNOW", "WHAT"]
CONFIDENCE_THRESHOLD = 0.8
SEQUENCE_LENGTH = 30
WS_HOST = "localhost"
WS_PORT = 8765

print("Loading action model...")
model = load_model("D:\\cl front\\action_model.h5")

base_options = mp_python.BaseOptions(model_asset_path="D:\\cl front\\hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)
print("Models loaded.")

def extract_keypoints(result):
    if result.hand_landmarks:
        hands = []
        for hand in result.hand_landmarks:
            for lm in hand:
                hands.extend([lm.x, lm.y, lm.z])
        while len(hands) < 126:
            hands.extend([0, 0, 0])
        return np.array(hands[:126])
    return np.zeros(126)
async def improve_sentence_with_gemini(words):

    text = " ".join(words)

    prompt = f"""
    Rewrite this into a natural grammatically correct sentence.

    Keep meaning same.
    Do not explain.
    Return only sentence.

    Input: {text}
    """

    try:

        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt
        )

        return response.text.strip()

    except Exception as e:

        print("Gemini error:", e)

        return "Gemini unavailable"
async def detection_handler(websocket):

    print(f"Client connected: {websocket.remote_address}")

    sequence = []

    try:

        async for message in websocket:

            try:

                if isinstance(message, str):

                    try:

                        data = json.loads(message)

                        if data.get("type") == "improve_sentence":

                            words = data.get("sentence", [])

                            improved = await improve_sentence_with_gemini(words)

                            await websocket.send(json.dumps({
                                "type": "gemini_result",
                                "text": improved
                            }))

                            continue

                    except:
                        pass

                encoded_data = message.split(',')[1]

                nparr = np.frombuffer(
                    base64.b64decode(encoded_data),
                    np.uint8
                )

                frame = cv2.imdecode(
                    nparr,
                    cv2.IMREAD_COLOR
                )

                if frame is None:
                    continue

                frame = cv2.flip(frame, 1)

                frame_rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                timestamp = int(time.time() * 1000)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=frame_rgb
                )

                result = detector.detect_for_video(
                    mp_image,
                    timestamp
                )

                if result.hand_landmarks:

                    kp = extract_keypoints(result)

                    sequence.append(kp)

                else:
                    sequence = []

                sequence = sequence[-SEQUENCE_LENGTH:]

                payload = {
                    "probs": [0.0] * len(ACTIONS),
                    "word": None,
                    "confidence": 0.0
                }

                if len(sequence) == SEQUENCE_LENGTH:

                    res = model.predict(
                        np.expand_dims(sequence, axis=0),
                        verbose=0
                    )[0]

                    probs = res.tolist()

                    max_idx = int(np.argmax(res))

                    confidence = float(np.max(res))

                    payload = {
                        "probs": probs,
                        "word": ACTIONS[max_idx]
                        if confidence > CONFIDENCE_THRESHOLD
                        else None,
                        "confidence": confidence
                    }

                await websocket.send(json.dumps(payload))

            except Exception as e:

                print("Frame processing error:", e)

    except websockets.exceptions.ConnectionClosed:

        print("Client disconnected")


async def main():
    print(f"Starting TTMH WebSocket server on ws://{WS_HOST}:{WS_PORT}")
    async with websockets.serve(detection_handler, WS_HOST, WS_PORT):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
