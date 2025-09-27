# Face Recognition System 

## 📌 Overview
This project implements a **real-time face recognition system** that can be trained on-the-fly using your webcam and tested immediately.  
The system uses **deep learning embeddings** to identify and verify faces with high accuracy.

---

## 📂 Project Structure
face-recognition-system/
├── train_face_model.py # Training script
├── test_face_model.py # Testing script
├── README.md # Project documentation
└── my_faces/ # Folder for training images (auto-created)

---

## ✨ Features
- **Video-Based Training**: Collect face embeddings from live camera feed  
- **Real-Time Recognition**: Instantly test the trained model  
- **Persistent Model Storage**: Save/load models for future use  
- **Cross-Platform Compatibility**: Works on Windows/macOS/Linux  

---

## 🛠️ Installation Requirements
```bash
pip install opencv-python deepface numpy
________________________________________
🚀 Usage Instructions
1️ Training the Model
Run the training script:
python train_face_model.py
•	Position your face in front of the camera
•	Training runs for 30 seconds (customize with training_duration)
•	Press 'q' to stop early
•	Model is automatically saved to face_model.pkl
________________________________________
2️ Testing the Model
Run the testing script:
python test_face_model.py
•	Displays real-time recognition results
•	Press 'q' to exit
________________________________________
⚙️ Technical Details
•	Model Architecture: DeepFace (Facenet model by default)
•	Embedding Dimension: 512 vector per face
•	Similarity Metric: Cosine similarity
•	Performance: ~10–15 FPS, best at 640x480 resolution
________________________________________
🔧 Customization
•	Change training duration: training_duration in train_face_model.py
•	Adjust similarity threshold: similarity_threshold in both scripts
•	Switch models: set model_name in DeepFace calls ("VGG-Face", "ArcFace", etc.)
________________________________________
🛠️ Troubleshooting
•	Camera Not Detected: Close other apps using the webcam
•	Poor Recognition: Add more training samples or increase duration
•	False Positives: Raise similarity threshold
•	False Negatives: Lower similarity threshold or add more embeddings
