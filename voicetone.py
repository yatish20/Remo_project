import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load and preprocess audio data
def preprocess_audio(file_path, sr=22050, duration=3):
    """
    Load and preprocess audio file.

    Parameters:
        file_path (str): Path to the audio file.
        sr (int): Sampling rate.
        duration (int): Duration of audio in seconds (default: 3 seconds).

    Returns:
        np.array: MFCC features of the audio.
    """
    try:
        # Load audio file
        audio, sample_rate = librosa.load(file_path, sr=sr, duration=duration)
        
        # Ensure consistent audio length
        if len(audio) < sr * duration:
            pad_width = sr * duration - len(audio)
            audio = np.pad(audio, (0, pad_width))
        elif len(audio) > sr * duration:
            audio = audio[: sr * duration]

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfcc = np.mean(mfcc.T, axis=0)

        return mfcc

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Load and process dataset
def load_dataset(dataset_path):
    """
    Load and process dataset of audio files.

    Parameters:
        dataset_path (str): Path to the dataset directory.

    Returns:
        X (list): List of feature arrays.
        y (list): List of labels.
    """
    X, y = [], []
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            for file_name in os.listdir(label_path):
                file_path = os.path.join(label_path, file_name)
                features = preprocess_audio(file_path)
                if features is not None:
                    X.append(features)
                    y.append(label)
    return np.array(X), np.array(y)

# Path to dataset
dataset_path = "C:/Users/yatis/OneDrive/Desktop/Remo-project/voicedataset/audio_speech_actors_01-24"

# Load dataset
X, y = load_dataset(dataset_path)
print(f"Loaded dataset with {len(X)} samples.")

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode the labels to integers
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Initialize the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
print("Training the model...")
model.fit(X_train, y_train)
print("Model training complete!")

# Make predictions on the test data
print("Making predictions...")
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
