# Talk-To-My-Hands - Real-Time Sign Language Recognition System

## Project Overview

Talk-To-My-Hands is a real-time sign language recognition system developed using Deep Learning, MediaPipe, TensorFlow, OpenCV, and WebSockets.

The project detects hand movements using MediaPipe hand landmarks and predicts sign language gestures using an LSTM deep learning model.

The system also integrates Gemini AI to convert predicted words into grammatically correct natural sentences.

This project was developed as a college-level AI and Computer Vision project.

---

# Developed By

## Aryan Kanas

LinkedIn: [https://www.linkedin.com/](https://www.linkedin.com/)

## Hargunneet Kaur

LinkedIn: [https://www.linkedin.com/](https://www.linkedin.com/)

---

# Features

* Real-time sign language recognition
* Hand tracking using MediaPipe
* LSTM-based gesture recognition
* WebSocket-based frontend-backend communication
* Live webcam detection
* Confidence score display
* Sentence formation from detected gestures
* Gemini AI integration for grammar correction
* Responsive frontend UI
* Real-time prediction updates

---

# Technologies Used

## Frontend

* HTML
* CSS
* JavaScript
* WebSockets

## Backend

* Python
* TensorFlow / Keras
* OpenCV
* MediaPipe
* NumPy
* AsyncIO
* WebSockets

## AI Models

* LSTM Neural Network
* Gemini AI API

---

# Dataset Information

The project was initially inspired using the ASL Citizen dataset structure.

However, the final training data used in this project is a completely custom dataset created manually for the required gesture classes.

Custom gestures used:

* YOU
* ME
* WANT
* NEED
* KNOW
* WHAT

The dataset contains MediaPipe hand landmark sequences saved in `.npy` format.

---

# Project Structure

```bash
PROJECT/
│
├── data/
│   ├── YOU/
│   ├── ME/
│   ├── WANT/
│   ├── NEED/
│   ├── KNOW/
│   └── WHAT/
│
├── models/
│   └── hand_landmarker.task
│
├── frontend/
│   ├── index.html
│   
│
├── ws_server.py
├── train_model.py
├── collect_data.py
├── action_model.h5
├── requirements.txt
└── README.md
```

---

# Python Version Requirement

This project works properly with:

```bash
Python 3.11
```

TensorFlow currently has compatibility issues with Python 3.14 on Windows.

---

# Installation Guide

## 1. Clone or Download Project

Place the project folder anywhere on your system.

Example:

```bash
D:\PROJECT
```

---

# 2. Create Virtual Environment

Open PowerShell inside project folder:

```bash
python -m venv venv
```

---

# 3. Activate Virtual Environment

## PowerShell

```bash
venv\Scripts\activate
```

If execution policy error occurs:

```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again:

```bash
venv\Scripts\activate
```

---

# 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 5. Install Required Additional Libraries

```bash
pip install google-genai
pip install python-dotenv
pip install websockets
```

---

# 6. Gemini API Setup

Create a `.env` file in the root project folder.

Add:

```env
GEMINI_API_KEY=your_api_key_here
```

The Gemini API key is used for sentence improvement.

---

# 7. Download MediaPipe Hand Model

Download:

```bash
hand_landmarker.task
```

Place it inside:

```bash
models/
```

---

# Training the Model

## 1. Collect Landmark Data

Run:

```bash
python collect_data.py
```

This extracts hand landmarks using MediaPipe and saves `.npy` sequences.

---

## 2. Train LSTM Model

Run:

```bash
python train_model.py
```

After training:

```bash
action_model.h5
```

will be generated.

---

# Running the Project

## 1. Start Backend WebSocket Server

```bash
python ws_server.py
```

Backend runs on:

```bash
ws://localhost:8765
```

---

## 2. Start Frontend Server

Inside frontend folder:

```bash
python -m http.server 5500
```

Open browser:

```bash
http://localhost:5500
```

---

# How the System Works

```text
Frontend Webcam
        ↓
Canvas Frame Capture
        ↓
WebSocket Transfer
        ↓
Python Backend
        ↓
MediaPipe Hand Detection
        ↓
Landmark Extraction
        ↓
LSTM Prediction
        ↓
Frontend Display
        ↓
Gemini Sentence Improvement
```

---

# Model Architecture

The project uses an LSTM-based deep learning architecture.

## Architecture

```python
LSTM(64)
LSTM(128)
Dense(64, relu)
Dense(32, relu)
Dense(Output, softmax)
```

## Why LSTM?

LSTM works well for sequential data and gesture movement recognition because it remembers temporal patterns across frames.

---

# MediaPipe Hand Tracking

MediaPipe extracts:

* 21 hand landmarks
* x, y, z coordinates
* real-time hand tracking

Two-hand tracking gives:

```text
126 features per frame
```

Sequence size:

```text
30 frames
```

Final input shape:

```text
(30, 126)
```

---

# WebSocket Communication

WebSockets are used for real-time communication between frontend and backend.

Frontend sends:

* webcam frames

Backend sends:

* prediction probabilities
* predicted word
* confidence score
* Gemini corrected sentence

---

# Gemini AI Integration

Gemini AI improves detected words into natural grammatical sentences.

Example:

```text
Input:
ME WANT FOOD

Output:
I want food.
```

---

# Future Improvements

* More sign language gestures
* Sentence-level recognition
* Better UI animations
* Multi-language support
* Speech synthesis
* Mobile deployment
* Transformer-based models

---

# Conclusion

Talk-To-My-Hands demonstrates how Deep Learning, Computer Vision, and Real-Time Communication can be combined to build an intelligent sign language recognition system.

The project focuses on practical implementation, real-time performance, and accessibility using modern AI technologies.

---

# License

This project is developed for educational and academic purposes.
