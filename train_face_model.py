import cv2
import numpy as np
import os
from deepface import DeepFace
import threading
import time
import pickle

class FaceTrainer:
    def __init__(self):
        self.camera_index = 0
        self.face_folder = "my_faces/"
        self.model_file = "face_model.pkl"
        self.similarity_threshold = 0.6
        self.training_duration = 30  # Training time in seconds
        self.embeddings = []
        self.cap = None
        self.training_thread = None
        self.stop_training = False
        self.training_complete = False
        
        # Create face folder if needed
        if not os.path.exists(self.face_folder):
            os.makedirs(self.face_folder)
    
    def start_camera(self):
        """Initialize camera"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print("Camera not accessible. Check permissions/index.")
            exit(1)
    
    def collect_embeddings(self):
        """Collect face embeddings from real-time video"""
        start_time = time.time()
        count = 0
        
        while not self.stop_training and (time.time() - start_time) < self.training_duration:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(
                gray, 1.1, 5
            )
            
            for (x, y, w, h) in faces:
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Extract face ROI
                face_roi = frame[y:y+h, x:x+w]
                
                try:
                    # Get embedding
                    embedding = DeepFace.represent(face_roi, model_name="Facenet", enforce_detection=True)[0]["embedding"]
                    
                    # Save to memory
                    self.embeddings.append(embedding)
                    count += 1
                    
                    # Visual feedback
                    cv2.putText(frame, f"Training: {count} samples", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                except Exception as e:
                    print(f"Training error: {str(e)}")
            
            # Show training progress
            cv2.imshow('Training', frame)
            if cv2.waitKey(1) == ord('q'):
                self.stop_training = True
                break
        
        print(f"Training completed! Collected {len(self.embeddings)} face embeddings.")
        self.training_complete = True
    
    def save_model(self):
        """Save embeddings to file"""
        try:
            with open(self.model_file, 'wb') as f:
                pickle.dump(self.embeddings, f)
            print(f"Model saved to {self.model_file}")
        except Exception as e:
            print(f"Could not save model: {str(e)}")
    
    def run(self):
        """Main training workflow"""
        self.start_camera()
        self.collect_embeddings()
        self.save_model()
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    trainer = FaceTrainer()
    trainer.run()