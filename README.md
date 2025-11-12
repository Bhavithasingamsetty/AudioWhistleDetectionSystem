# AudioWhistleDetectionSystem
Applying Machine Learning to Detect and Count Cooker Whistles in Real Time
# Overview
AudioWhistle Detection System is a web-based project that detects cooker whistle sounds using a trained machine learning model.
The system listens to live audio input, counts the number of whistles, and provides a voice and desktop notification when a user-defined target is reached.
This project combines audio signal processing, machine learning, and web technologies to demonstrate how small, everyday problems can be solved with AI.
# Key Features
Detects cooker whistles using microphone input
Allows users to set a target whistle count through the web interface
Triggers a voice alert and browser notification when the target is reached
Real-time processing through Flask backend and JavaScript frontend
# Tech Stack
Programming: Python
Libraries: Librosa, Scikit-learn, NumPy, Flask
Frontend: HTML, CSS, JavaScript
Model: Random Forest Classifier
Environment	:Localhost(Flask Server)
# How It Works
Collects and labels audio samples into Whistle and Noise categories.
Extracts MFCC (Mel-Frequency Cepstral Coefficients) and other spectral features using Librosa.
Trains a Random Forest Classifier to differentiate between whistle and non-whistle sounds.
Saves the trained model as whistle_model.pkl.
The Flask backend uses this model for real-time classification of audio captured from the microphone.
The web interface updates the whistle count dynamically and issues an alert once the target is reached.
# Project Structure
AudioWhistleDetectionSystem/
│
├── data/
│   ├── Whistle/
│   ├── Noise/
│
├── model_training.py         # Extracts features and trains Random Forest model
├── whistle_model.pkl         # Saved trained model
├── app.py                    # Flask app for real-time detection
├── templates/
│   └── index.html            # Frontend UI
├── static/
│   ├── css/
│   └── js/
└── README.md
# Setup Instructions
1. Clone the repository
git clone https://github.com/Bhavithasingamsetty/AudioWhistleDetectionSystem.git
cd AudioWhistleDetectionSystem
2. Create a virtual environment
python3 -m venv env
source env/bin/activate   # For Mac/Linux
env\Scripts\activate      # For Windows
3. Install dependencies
pip install -r requirements.txt
4. Train the model (optional)
If you want to retrain the model:
python model_training.py
5. Run the Flask application
python app.py
Then open your browser and go to:
http://127.0.0.1:5000

Example Output
Whistle detected! Total: 1
Whistle detected! Total: 2
Target reached — stopping detection.
Final count: 2/2
Frontend display:
Whistles Detected: 2 / Target: 2
Target of 2 whistles reached!
# Future Improvements
Improve model accuracy using deep learning (CNN for spectrogram classification)
Add noise filtering and auto-calibration for different environments
Integrate IoT support to automatically switch off gas when the target is reached
Build mobile app version
# Author
Bhavitha Singamsetty
