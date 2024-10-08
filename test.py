import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=5)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

def load_dataset(cry_folder, non_cry_folder):
    features = []
    labels = []
    
    # Load cry files
    for filename in os.listdir(cry_folder):
        if filename.endswith('.wav'):  # Assuming .wav files, adjust if needed
            file_path = os.path.join(cry_folder, filename)
            features.append(extract_features(file_path))
            labels.append(1)
    
    # Load non-cry files
    for filename in os.listdir(non_cry_folder):
        if filename.endswith('.wav'):  # Assuming .wav files, adjust if needed
            file_path = os.path.join(non_cry_folder, filename)
            features.append(extract_features(file_path))
            labels.append(0)
    
    return np.array(features), np.array(labels)

def create_model():
    model = Sequential([
        LSTM(64, input_shape=(40, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Specify the folders containing cry and non-cry audio files
cry_folder = r"C:\Users\Asus\Downloads\baby_cry_detection-master\Baby cry\baby_cry_detection-master\data\301 - Crying baby"
non_cry_folder = r"C:\Users\Asus\Downloads\baby_cry_detection-master\Baby cry\baby_cry_detection-master\data\901 - Silence"
# Load and preprocess the data
X, y = load_dataset(cry_folder, non_cry_folder)
X = StandardScaler().fit_transform(X)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = create_model()
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy}")

# Function to predict on new audio
def predict_cry(file_path, model):
    features = extract_features(file_path)
    features = StandardScaler().fit_transform(features.reshape(1, -1))
    features = features.reshape(1, 40, 1)
    prediction = model.predict(features)
    return prediction[0][0]

# Example usage
new_audio = r"C:\Users\Asus\Downloads\baby-laughing-02.wav"
cry_probability = predict_cry(new_audio, model)
print(f"Probability of baby cry: {cry_probability}")