import cv2
import numpy as np
import os
from deepface import DeepFace
import threading
import time
import pickle

class FaceTester:
    def __init__(self):
        self.camera_index = 0
        self.model_file = "face_model.pkl"
        self.similarity_threshold = 0.6
        self.embeddings = []
        self.cap = None
        
        # Load existing model
        self.load_model()
    
    def load_model(self):
        """Load saved embeddings from file"""
        try:
            with open(self.model_file, 'rb') as f:
                self.embeddings = pickle.load(f)
            print(f"Loaded {len(self.embeddings)} embeddings from {self.model_file}")
        except Exception as e:
            print(f"Could not load model: {str(e)}. Exiting...")
            exit(1)
    
    def start_camera(self):
        """Initialize camera"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print("Camera not accessible. Check permissions/index.")
            exit(1)
    
    def recognize_faces(self):
        """Recognize faces in real-time using trained embeddings"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(
                gray, 1.1, 5
            )
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                face_roi = frame[y:y+h, x:x+w]
                
                try:
                    live_embedding = DeepFace.represent(face_roi, model_name="Facenet", enforce_detection=True)[0]["embedding"]
                    
                    # Compare with trained embeddings
                    best_match = False
                    for stored_embedding in self.embeddings:
                        similarity = np.dot(live_embedding, stored_embedding) / (
                            np.linalg.norm(live_embedding) * np.linalg.norm(stored_embedding)
                        )
                        if similarity > self.similarity_threshold:
                            best_match = True
                            break
                    
                    label = "YOUR FACE!" if best_match else "Unknown"
                    color = (0, 255, 0) if best_match else (0, 0, 255)
                    cv2.putText(frame, label, (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    
                except Exception as e:
                    print(f"Recognition error: {str(e)}")
            
            cv2.imshow('Recognition', frame)
            if cv2.waitKey(1) == ord('q'):
                break
    
    def run(self):
        """Main testing workflow"""
        self.start_camera()
        print("Starting recognition mode. Press 'q' to quit.")
        self.recognize_faces()
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tester = FaceTester()
    tester.run()