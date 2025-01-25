import cv2
import mediapipe as mp
from deepface import DeepFace
import time
import os
from pathlib import Path
from threading import Thread
import queue

# KONFİGÜRASYON AYARLARI
BASE_DIR = Path(__file__).parent
CONFIG = {
    "camera_index": 0,
    "emoji_path": str(BASE_DIR / "emojipng"),
    "resolution": (1280, 720),
    "emoji_size": (80, 80),
    "collection_pos": (30, 30),
    "min_detection_confidence": 0.75,
    "target_emotions": ["happy", "sad", "angry", "surprise", "neutral"],
    "game_font": cv2.FONT_HERSHEY_DUPLEX,
    "uncollected_mark": "-",
    "collected_mark": "+",
    "game_time": 30,
    "box_color": (0, 165, 255),
    "text_color": (255, 255, 255),
    "warning_color": (0, 0, 255),
    "restart_text": "New Game: 'R'\n Exit: 'Q'"
}

class EmotionGame:
    def __init__(self):
        self.cap = cv2.VideoCapture(CONFIG["camera_index"])
        self.set_camera_resolution()
        
        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        
        self.emojis = self.load_emojis()
        self.reset_game()
        
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=CONFIG["min_detection_confidence"]
        )
        
        Thread(target=self.analysis_thread, daemon=True).start()

    def reset_game(self):
        self.current_target_index = 0
        self.score = 0
        self.start_time = time.time()
        self.game_active = True
        self.game_finished = False
        self.end_time = None
        self.last_emotion = ""
        self.time_added = False
        self.clear_queues()

    def clear_queues(self):
        while not self.frame_queue.empty():
            try: self.frame_queue.get_nowait()
            except queue.Empty: break
        while not self.result_queue.empty():
            try: self.result_queue.get_nowait()
            except queue.Empty: break

    def set_camera_resolution(self):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["resolution"][0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["resolution"][1])

    def load_emojis(self):
        emojis = {}
        for emotion in CONFIG["target_emotions"]:
            emoji_path = Path(CONFIG["emoji_path"]) / f"{emotion}.png"
            img = cv2.imread(str(emoji_path), cv2.IMREAD_UNCHANGED)
            if img.shape[2] == 4:
                alpha = img[:,:,3]
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            emojis[emotion] = img
        return emojis

    def analysis_thread(self):
        while True:
            try:
                frame = self.frame_queue.get(timeout=1)
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                if result:
                    self.result_queue.put(result[0]["dominant_emotion"].lower())
            except Exception as e:
                print(f"Analiz hatası: {str(e)}")
                self.result_queue.put(None)

    def draw_collection(self, frame):
        x_start, y_start = CONFIG["collection_pos"]
        emoji_w, emoji_h = CONFIG["emoji_size"]
        
        for i in range(self.current_target_index + 1):
            if i >= len(CONFIG["target_emotions"]): break
            emotion = CONFIG["target_emotions"][i]
            y = y_start + i * (emoji_h + 15)
            
            cv2.rectangle(frame, (x_start, y), (x_start + emoji_w, y + emoji_h), (255, 255, 255), 2)
            resized = cv2.resize(self.emojis[emotion], CONFIG["emoji_size"])
            frame[y:y+emoji_h, x_start:x_start+emoji_w] = resized
            
            if i < self.current_target_index:
                mark, color = CONFIG["collected_mark"], (0, 255, 0)
            else:
                mark, color = CONFIG["uncollected_mark"], CONFIG["warning_color"]
                
            cv2.putText(frame, mark, (x_start + emoji_w + 15, y + emoji_h//2 + 5),
                       CONFIG["game_font"], 1.2, color, 3)

    def draw_face_info(self, frame, detection):
        image_height, image_width, _ = frame.shape
        box = detection.location_data.relative_bounding_box
        
        x = int(box.xmin * image_width)
        y = int(box.ymin * image_height)
        w = int(box.width * image_width)
        h = int(box.height * image_height)
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), CONFIG["box_color"], 3)
        
        text = f"{self.last_emotion.upper()}"
        (tw, th), _ = cv2.getTextSize(text, CONFIG["game_font"], 1.1, 2)
        cv2.rectangle(frame, (x, y - th - 20), (x + tw + 10, y - 10), CONFIG["box_color"], -1)
        cv2.putText(frame, text, (x + 5, y - 20), CONFIG["game_font"], 1.1, CONFIG["text_color"], 2)

    def update_game_state(self, emotion):
        emotion = emotion.lower()
        if self.game_active and not self.game_finished:
            if self.current_target_index < len(CONFIG["target_emotions"]):
                target_emotion = CONFIG["target_emotions"][self.current_target_index]
                if emotion == target_emotion:
                    self.current_target_index += 1
                    self.score += 20
                    if self.current_target_index == len(CONFIG["target_emotions"]):
                        # Kalan süreyi skora ekle
                        elapsed = time.time() - self.start_time
                        remaining = CONFIG["game_time"] - int(elapsed)
                        if remaining > 0 and not self.time_added:
                            self.score += remaining
                            self.time_added = True
                        self.game_active = False
                        self.game_finished = True
                        self.end_time = time.time()

    def draw_game_info(self, frame):
        # Zaman hesaplama
        if self.game_finished:
            elapsed = self.end_time - self.start_time
            remaining = max(CONFIG["game_time"] - int(elapsed), 0)
        else:
            elapsed = time.time() - self.start_time
            remaining = max(CONFIG["game_time"] - int(elapsed), 0)
            if remaining <= 0:
                self.game_active = False
                self.game_finished = True
                self.end_time = time.time()
                remaining = 0

        # Zaman çubuğu
        progress_width = int(700 * (remaining / CONFIG["game_time"]))
        cv2.rectangle(frame, (30, CONFIG["resolution"][1] - 50), 
                     (30 + progress_width, CONFIG["resolution"][1] - 30), (0, 255, 0), -1)
        
        # Skor ve zaman metinleri (aynı hizada)
        score_x = CONFIG["resolution"][0] - 250
        cv2.putText(frame, f"Score: {self.score}", (score_x, 50),
                   CONFIG["game_font"], 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Time: {remaining}s", (score_x, 100),
                   CONFIG["game_font"], 0.8, (255, 255, 0), 2)
        
        # Oyun sonu ekranı
        if self.game_finished:
            status_text = "Tebrikler!" if self.current_target_index == len(CONFIG["target_emotions"]) else "Time Over!!"
            text_size = cv2.getTextSize(status_text, CONFIG["game_font"], 1.8, 4)[0]
            text_x = (CONFIG["resolution"][0] - text_size[0]) // 2
            text_y = (CONFIG["resolution"][1] + text_size[1]) // 2
            cv2.putText(frame, status_text, (text_x, text_y), CONFIG["game_font"], 1.8, (0, 255, 255), 4)
            
            restart_size = cv2.getTextSize(CONFIG["restart_text"], CONFIG["game_font"], 0.8, 2)[0]
            restart_x = (CONFIG["resolution"][0] - restart_size[0]) // 2
            cv2.putText(frame, CONFIG["restart_text"], (restart_x, text_y + 60),
                       CONFIG["game_font"], 0.8, (0, 255, 0), 2)

    def process_frame(self, frame):
        if self.game_active and not self.game_finished:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)
            
            if results.detections:
                for detection in results.detections:
                    self.draw_face_info(frame, detection)
                
                try: self.frame_queue.put(frame.copy(), block=False)
                except queue.Full: pass
                
                while not self.result_queue.empty():
                    emotion = self.result_queue.get()
                    if emotion: 
                        self.last_emotion = emotion
                        self.update_game_state(emotion)
        return frame

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            frame = self.process_frame(frame)
            self.draw_collection(frame)
            self.draw_game_info(frame)
            
            cv2.imshow('Emoji Avi', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('r') and not self.game_active: self.reset_game()

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    game = EmotionGame()
    game.run()