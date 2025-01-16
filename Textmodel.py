import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load dataset
dataset_path = "goemotions_1.csv"  # Replace with the actual dataset path

try:
    # Load specific columns if required
    columns_to_load = ["text", "admiration", "amusement", "anger", "annoyance", 
                       "approval", "caring", "confusion", "curiosity", "desire", 
                       "disappointment", "disapproval", "disgust", "embarrassment", 
                       "excitement", "fear", "gratitude", "grief", "joy", "love", 
                       "nervousness", "optimism", "pride", "realization", "relief", 
                       "remorse", "sadness", "surprise", "neutral"]
    data = pd.read_csv(dataset_path, usecols=columns_to_load)

    print("Dataset loaded successfully!")
    print(data.head())  # Display the first few rows of the dataset

except FileNotFoundError:
    print(f"Error: File not found at {dataset_path}. Please check the file path.")
    exit()

# Extract text and emotion labels
texts = data['text'].tolist()  # Ensure 'text' column exists

# Combine multiple emotion columns into a single label for simplicity
# For this example, selecting the first emotion with a value of 1 per row
labels = data.drop(columns=['text']).idxmax(axis=1).tolist()

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(texts, encoded_labels, test_size=0.2, random_state=42)

# Tokenize text
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_length = 100
X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding="post", truncating="post")
X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding="post", truncating="post")

# Build model
model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=max_length),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
history = model.fit(
    X_train_padded, np.array(y_train),
    validation_data=(X_test_padded, np.array(y_test)),
    epochs=10,
    batch_size=32
)

# Evaluate model
loss, accuracy = model.evaluate(X_test_padded, np.array(y_test))
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the model and tokenizer
model.save("text_emotion_recognition_model.h5")
with open("tokenizer.json", "w") as f:
    f.write(tokenizer.to_json())

# Map emotions to genres
emotion_to_genre = {
    "joy": "comedy",
    "anger": "action",
    "sadness": "drama",
    "fear": "horror",
    "love": "romance",
    # Add more mappings based on your requirements
}

# Test model prediction
def predict_genre(input_text):
    sequence = tokenizer.texts_to_sequences([input_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding="post", truncating="post")
    prediction = model.predict(padded_sequence)
    predicted_emotion = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    return emotion_to_genre.get(predicted_emotion, "unknown")


# Example
input_text = "I am so happy today!"
recommended_genre = predict_genre(input_text)
print(f"Recommended Genre: {recommended_genre}")
