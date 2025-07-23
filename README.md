This project combines AI-powered proctoring with real-time facial expression analysis to monitor candidates during online interviews or assessments. It leverages computer vision and deep learning techniques to ensure exam integrity and evaluate candidate personality traits through emotional and behavioral cues.

🔍 Key Features

- 🧑‍💻 **AI Proctoring**:
  - Detects face presence, multiple persons, face spoofing, and suspicious behavior
  - Head pose estimation and eye tracking for attention monitoring
  - Mouth opening and face occlusion detection (e.g., mask or hand covering)

- 😃 **Real-Time Personality Evaluation**:
  - Emotion detection using CNN on live video feed
  - Maps emotion patterns to inferred personality traits
  - Generates visual summaries of emotional responses

- 📊 **Insights Dashboard**:
  - Real-time emotion overlay
  - Charts showing expression trends
  - Final personality analysis report

🧠 Technologies Used

- **Languages**: Python
- **Libraries**: OpenCV, Dlib, NumPy, TensorFlow/Keras, Matplotlib, DeepFace
- **Model**: Custom CNN trained on FER-2013 dataset for emotion classification
- **Tools**: Flask/Streamlit for UI, MediaPipe (optional for facial landmarks), Pandas

🗂️ Project Structure

├── app.py # Main application script
├── proctoring/
│ ├── face_detection.py # Face/eye/mouth tracking logic
│ └── spoof_detection.py # Spoof attack detection
├── evaluation/
│ ├── emotion_model.h5 # Pretrained emotion CNN model
│ └── emotion_analysis.py # Personality mapping logic
├── templates/
│ └── index.html # Web UI
├── static/
│ └── styles.css # Optional CSS
├── utils/
│ └── utils.py # Helper functions
├── requirements.txt
└── README.md

🚀 How to Run
git clone https://github.com/Pallavi-2042/ai-proctoring-interview-evaluation.git
cd ai-proctoring-interview-evaluation
pip install -r requirements.txt
python app.py 
Make sure your webcam is enabled and accessible by the application.

📌 Use Cases
Online interviews and assessments
University placement training systems
Behavioral research and HR analytics
Fraud detection during remote exams

🛠️ Future Scope
Speech sentiment analysis with emotion
Candidate scoring system based on behavioral patterns
Admin dashboard to track multiple sessions
