import cv2
import numpy as np
import time
import os
import random
import pygame
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from collections import Counter
from typing import Dict, List

MUSIC = True

def load_gif(directory: str) -> Dict[str, List[np.ndarray]]:
    gif_dict = {emotion: [] for emotion in get_emotion_labels()}
    for emotion in gif_dict.keys():
        emotion_folder = os.path.join(directory, emotion)
        if os.path.exists(emotion_folder):
            for file in os.listdir(emotion_folder):
                if file.endswith('.gif'):
                    gif_path = os.path.join(emotion_folder, file)
                    gif = cv2.VideoCapture(gif_path)
                    frames = []
                    while True:
                        ret, frame = gif.read()
                        if not ret:
                            break
                        frames.append(frame)
                    gif_dict[emotion].append(frames)
    return gif_dict

def load_emotion_model(model_path='face_model.weights.h5'):
    return load_model(model_path)

def get_emotion_labels():
    return ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def initialize_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def capture_video():
    return cv2.VideoCapture(0)

def load_music() -> Dict[str, tuple]:
    music_files = {
        'Angry': ('musics/happy.mp3', 30),
        'Disgusted': ('musics/happy.mp3', 30),
        'Fear': ('musics/happy.mp3', 30),
        'Happy': ('musics/happy.mp3', 30),
        'Sad': ('musics/happy.mp3', 30),
        'Surprise': ('musics/happy.mp3', 30)
    }
    return music_files

def process_faces(frame, face_cascade, model, class_names, emotion_window):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]
        face_image = cv2.resize(face_roi, (48, 48))
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = image.img_to_array(face_image)
        face_image = np.expand_dims(face_image, axis=0)
        face_image = np.vstack([face_image])

        predictions = model.predict(face_image)
        emotion_label = class_names[np.argmax(predictions)]
        emotion_window.append(emotion_label)
        cv2.putText(frame, f'Emotion: {emotion_label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('Emotion Detection', frame)

def display_gif(emotion, gif_dict):
    if emotion in gif_dict and gif_dict[emotion]:
        gif_frames = random.choice(gif_dict[emotion])
        for frame in gif_frames:
            cv2.imshow('Emotion Detection', frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

def play_music(emotion, music_files):
    global MUSIC
    if not MUSIC:
        return
    MUSIC = False
    file_path, start_pos = music_files.get(emotion, (None, 0))
    if file_path:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play(start=0, fade_ms=1000)
        pygame.mixer.music.set_pos(start_pos)

def display_blue_screen(emotion_window, gif_dict, music_files):
    if emotion_window:
        most_common_emotion = Counter(emotion_window).most_common(1)[0][0]
    else:
        most_common_emotion = "Neutral"

    play_music(most_common_emotion, music_files)
    display_gif(most_common_emotion, gif_dict)


def analyze_emotions(face_cascade, model, class_names, emotion_window, gif_dict, music_files):
    global MUSIC
    cap = capture_video()
    start_time = time.time()
    cycle_duration = 15  # 5 secondes pour l'analyse, 10 secondes pour l'écran bleu

    while True:
        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time <= 5:
            ret, frame = cap.read()
            process_faces(frame, face_cascade, model, class_names, emotion_window)
        else:
            display_blue_screen(emotion_window, gif_dict, music_files)

            if elapsed_time >= cycle_duration:
                start_time = time.time()
                emotion_window.clear()
                MUSIC = True

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    pygame.mixer.init()
    model = load_emotion_model()
    class_names = get_emotion_labels()
    face_cascade = initialize_face_cascade()
    emotion_window = []
    gif_dict = load_gif('gifs')
    music_files = load_music()
    analyze_emotions(face_cascade, model, class_names, emotion_window, gif_dict, music_files)

if __name__ == "__main__":
    main()
