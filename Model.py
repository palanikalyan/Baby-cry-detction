import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical
import tkinter as tk
from tkinter import filedialog
import joblib

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, duration=5)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def load_dataset(base_folder):
    features = []
    labels = []
    class_mapping = {'301 - Crying baby': 0, '901 - Silence': 1, '902 - Noise': 2, '903 - Baby laugh': 3}
    
    for class_name in class_mapping:
        class_folder = os.path.join(base_folder, class_name)
        if not os.path.exists(class_folder):
            print(f"Warning: Folder not found - {class_folder}")
            continue
        
        for filename in os.listdir(class_folder):
            if filename.endswith('.wav'):
                file_path = os.path.join(class_folder, filename)
                feature = extract_features(file_path)
                if feature is not None:
                    features.append(feature)
                    labels.append(class_mapping[class_name])
    
    if not features:
        raise ValueError("No valid audio files were found. Please check your dataset.")
    
    return np.array(features), np.array(labels)

def create_model(input_shape, num_classes):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Set up the base folder path
base_folder = r"C:\Users\Asus\Downloads\baby_cry_detection-master\Baby cry\baby_cry_detection-master\data"
model_path = 'baby_cry_model.h5'
scaler_path = 'baby_cry_scaler.joblib'

def train_or_load_model():
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print("Loading existing model and scaler...")
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler

    print("Training new model...")
    # Load and preprocess the data
    print("Loading dataset...")
    X, y = load_dataset(base_folder)
    print(f"Loaded {len(X)} samples.")

    # Create and fit the scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

    # Convert labels to one-hot encoded format
    y = to_categorical(y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Create and train the model
    num_classes = y.shape[1]
    input_shape = (X_scaled.shape[1], 1)
    model = create_model(input_shape, num_classes)
    print("Training model...")
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

    # Evaluate the model
    print("Evaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy}")

    # Save the model and scaler
    print("Saving model and scaler...")
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    return model, scaler

# Function to predict on new audio
def predict_cry(file_path, model, scaler):
    features = extract_features(file_path)
    if features is None:
        return None
    
    features_scaled = scaler.transform(features.reshape(1, -1))
    features_scaled = features_scaled.reshape(1, features_scaled.shape[1], 1)
    
    prediction = model.predict(features_scaled)
    return prediction[0]

try:
    # Train or load the model
    model, scaler = train_or_load_model()

    # Set up file dialog
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    while True:
        print("\nSelect a WAV file for prediction (or cancel to exit):")
        new_audio = filedialog.askopenfilename(
            initialdir=base_folder,
            title="Select WAV file for prediction",
            filetypes=(("WAV files", "*.wav"), ("all files", "*.*"))
        )
        
        if not new_audio:  # User cancelled
            break
        
        print(f"Processing file: {new_audio}")
        probabilities = predict_cry(new_audio, model, scaler)
        if probabilities is not None:
            class_names = ['Crying baby', 'Silence', 'Noise', 'Baby laugh']
            for class_name, prob in zip(class_names, probabilities):
                print(f"Probability of {class_name}: {prob:.4f}")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    print("Please check your dataset folder structure and file paths.")