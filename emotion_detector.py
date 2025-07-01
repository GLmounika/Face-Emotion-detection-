import cv2 
from deepface import DeepFace 
import os
import numpy as np

def load_emojis(folder_path):
    emojis = {}
    emotions = ['happy', 'sad', 'angry', 'surprise', 'disgust', 'neutral', 'fear']
    for emotion in emotions:
        path = os.path.join(folder_path, f"{emotion}.png")
        if os.path.exists(path):
            emoji_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if emoji_img is not None:
                if emoji_img.shape[2] == 3:
                    emoji_img = cv2.cvtColor(emoji_img, cv2.COLOR_BGR2BGRA)
                emojis[emotion] = emoji_img
            else:
                print(f"Failed to load {emotion}.png")
        else:
            print(f"Emoji not found: {path}")
    return emojis

def overlay_emoji(frame, emoji, x=10, y=10):
    try:
        emoji = cv2.resize(emoji, (100, 100))
        for i in range(emoji.shape[0]):
            for j in range(emoji.shape[1]):
                if emoji[i, j][3] != 0:
                    if y+i < frame.shape[0] and x+j < frame.shape[1]:
                        frame[y+i, x+j] = emoji[i, j][:3]
    except Exception as e:
        print("Overlay error:", e)

emotion_map = {
    'happy': 'happy',
    'sad': 'sad',
    'angry': 'angry',
    'surprise': 'surprise',
    'disgust': 'disgust',
    'neutral': 'neutral',
    'fear': 'fear',
    'fearful': 'fear',
    'surprised': 'surprise',
    'calm': 'neutral',
    'confused': 'neutral',
    'disgusted': 'disgust',
    'excited': 'happy'
}

emojis = load_emojis("emojis")  # Ensure emojis folder exists and contains .pngs

cap = cv2.VideoCapture(0)

frame_count = 0
last_emotion = "neutral"
emotion_update_interval = 30

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % emotion_update_interval == 0:
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            full_emotion = result[0]['dominant_emotion']
            mapped_emotion = emotion_map.get(full_emotion.lower(), 'neutral')
            print(f"Detected: {full_emotion} → Using: {mapped_emotion}")
            if mapped_emotion in emojis:
                last_emotion = mapped_emotion
        except Exception as e:
            print("Emotion detection error:", e)

    if last_emotion in emojis:
        overlay_emoji(frame, emojis[last_emotion])

    cv2.imshow("Emotion Detector", frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
