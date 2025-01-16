# Import Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Load Dataset
# Replace 'fer2013.csv' with the path to your FER-2013 dataset
data = pd.read_csv('fer2013.csv')


# Load the large CSV file


# Split the file into smaller chunks
chunk_size = 100000  # You can adjust this size
for i in range(0, data.shape[0], chunk_size):
    chunk = data.iloc[i:i+chunk_size]
    chunk.to_csv(f'fer2013_part_{i // chunk_size + 1}.csv', index=False)

# Preprocess the Data
def preprocess_data(data):
    images = data['pixels'].apply(lambda x: np.array(x.split(), dtype='float32').reshape(48, 48, 1) / 255.0)
    images = np.stack(images.values)  # Stack into a single array
    labels = tf.keras.utils.to_categorical(data['emotion'], num_classes=7)  # One-hot encode labels
    return images, labels

images, labels = preprocess_data(data)

# Split into Training and Validation Sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build the CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotions
])

# Compile the Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(X_train)

# Train the Model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    validation_data=(X_val, y_val),
    epochs=30,
    verbose=1
)

# Save the Model
model.save('emotion_detection_model.h5')

# Evaluate the Model
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Accuracy: {val_accuracy:.2f}")
