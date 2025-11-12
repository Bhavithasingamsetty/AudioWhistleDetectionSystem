import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Folders
base_dir = "data"
categories = ["Whistle", "Noise"]

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print("Error:", file_path, e)
        return None

X, y = [], []

for label, category in enumerate(categories):
    folder = os.path.join(base_dir, category)
    for filename in os.listdir(folder):
        if filename.endswith(".wav"):
            path = os.path.join(folder, filename)
            features = extract_features(path)
            if features is not None:
                X.append(features)
                y.append(label)

X, y = np.array(X), np.array(y)
print(f"Extracted features from {len(X)} files")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(f"Accuracy: {acc*100:.2f}%")

with open("whistle_model.pkl", "wb") as f:
    pickle.dump(model, f)

print(" Model saved as whistle_model.pkl")
