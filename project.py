import os
import cv2
import mediapipe as mp
import openai
import requests
import base64
import pyttsx3
import time
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# MediaPipe 手部模型初始化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 初始化 pyttsx3
engine = pyttsx3.init()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def guess_image(file_path, prompt):
    base64_image = encode_image(file_path)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']

def speak_text(text):
    engine.say(text)
    engine.runAndWait()

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
gesture_detected = False
start_time = 0
stable_time = 0
pinch_type = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 翻轉圖像以進行自我視圖
    frame = cv2.flip(frame, 1)

    # 將圖像從 BGR 轉換為 RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 獲取手指尖點
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # 計算各個手指與拇指之間的距離
            index_distance = ((thumb_tip.x - index_finger_tip.x) ** 2 + (thumb_tip.y - index_finger_tip.y) ** 2) ** 0.5
            middle_distance = ((thumb_tip.x - middle_finger_tip.x) ** 2 + (thumb_tip.y - middle_finger_tip.y) ** 2) ** 0.5
            ring_distance = ((thumb_tip.x - ring_finger_tip.x) ** 2 + (thumb_tip.y - ring_finger_tip.y) ** 2) ** 0.5
            pinky_distance = ((thumb_tip.x - pinky_tip.x) ** 2 + (thumb_tip.y - pinky_tip.y) ** 2) ** 0.5

            # 設置閾值來檢測捏合手勢
            threshold = 0.05
            if index_distance < threshold:
                pinch_type = "index"
                stable_time += 1
            elif middle_distance < threshold:
                pinch_type = "middle"
                stable_time += 1
            elif ring_distance < threshold:
                pinch_type = "ring"
                stable_time += 1
            elif pinky_distance < threshold:
                pinch_type = "pinky"
                stable_time += 1
            else:
                stable_time = 0
                pinch_type = ""

            if stable_time >= 10:  # 假設每秒檢測10次，10次穩定相當於1秒
                gesture_detected = True
                stable_time = 0

            cv2.putText(frame, f'Pinching: {pinch_type}', (10, 50), font, 1, (0, 0, 0), 2)

    if gesture_detected:
        # Capture the image after the delay
        ret, capture = cap.read()
        if ret:
            file_path = 'capture.jpg'
            cv2.imwrite(file_path, capture)

            # 使用根據捏合手指類型的 prompt 描述圖像
            if pinch_type == "index":
                prompt = "describe the color of the object i'm holding in one or two word. just the color"
            elif pinch_type == "middle":
                prompt = "describe what the object is i'm holding in one or two words"
            elif pinch_type == "ring":
                prompt = "describe the texture of the object i'm holding in one or two words"
            elif pinch_type == "pinky":
                prompt = "guess the size of the object i'm holding in one or two words in cm"
            else:
                prompt = "guess the content of the image."

            description = guess_image(file_path, prompt)
            cv2.putText(frame, description, (10, 180), font, 1, (0, 0, 255), 2)
            speak_text(description)  # 語音朗讀描述
            cv2.imshow('Camera', frame)
            cv2.waitKey(2000)  # 顯示結果2秒
            gesture_detected = False  # 重置手勢檢測狀態

    # 添加手勢-功能列表到右下角
    gesture_function_list = [
        "Index pinch: object color",
        "Middle pinch: object name",
        "Ring pinch: object texture",
        "Pinky pinch: object size"
    ]
    y0, dy = frame.shape[0] - 120, 30  # 底部位置和行間距

    for i, line in enumerate(gesture_function_list):
        y = y0 + i * dy
        cv2.putText(frame, line, (frame.shape[1] - 400, y), font, 0.7, (0, 255, 0), 2)

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
