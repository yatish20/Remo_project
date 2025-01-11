# Emotion-Based Entertainment Assistant

## Project Overview
The **Emotion-Based Entertainment Assistant** is an AI-driven system designed to analyze real-time emotions using facial expressions, voice tone, and typing styles. Based on the user's emotions, the assistant recommends personalized content such as movies, TV shows, music, or podcasts to match or uplift the user's mood. It also features a 'Mood Shifter' to help users transition to a desired emotional state.

## Features
- **Emotion Detection**: Analyze emotions using facial expressions, voice tone, and typing behavior.
- **Personalized Recommendations**: Suggest content tailored to the user's mood.
- **Mood Shifter**: Offer activities or content to help users shift their mood.
- **Smart Device Integration**: Work seamlessly with smart devices for a better user experience.
- **Custom Playlists**: Automatically create playlists based on user preferences and current emotions.

---

## Project Structure

### Backend
- **Technology**: Node.js, Express
- **Purpose**: API for emotion detection, recommendations, and integration with AI models.
- **Directory Structure**:
  ```
  backend/
    |-- app.js
    |-- routes/
    |-- controllers/
    |-- models/
    |-- services/
    |-- config/
  ```

### Frontend
- **Technology**: React.js
- **Purpose**: User interface for emotion detection and content recommendations.
- **Directory Structure**:
  ```
  frontend/
    |-- src/
        |-- components/
        |-- pages/
        |-- services/
        |-- styles/
  ```

### AI/ML Component
- **Technology**: Python (TensorFlow/Keras, OpenCV, and Librosa)
- **Purpose**: Emotion detection using facial expression, voice tone, and typing style.
- **Directory Structure**:
  ```
  ai-models/
    |-- facemodel.py
    |-- voicemodel.py
    |-- typingmodel.py
    |-- utils/
    |-- data/
  ```

---

## Phase 1: Emotion Detection

### Implementation
1. **Facial Expression Analysis**:
   - Use `fer2013` dataset for training.
   - Build a Convolutional Neural Network (CNN) for classifying emotions.
   - Tools: TensorFlow/Keras, OpenCV.

2. **Voice Tone Analysis**:
   - Analyze voice pitch, tone, and tempo.
   - Dataset: `RAVDESS` (Ryerson Audio-Visual Database of Emotional Speech and Song).
   - Tools: Librosa, TensorFlow.

3. **Typing Style Analysis**:
   - Monitor typing speed, key presses, and patterns.
   - Tools: Python, custom algorithms.

---

## Setup Instructions

### Prerequisites
- **Backend**:
  - Node.js (v16+)
  - MySQL
- **Frontend**:
  - Node.js (v16+)
  - React (v18+)
- **AI/ML**:
  - Python (v3.10+)
  - Libraries: TensorFlow, Keras, OpenCV, Librosa, Pandas

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/emotion-based-assistant.git
   cd emotion-based-assistant
   ```

2. **Setup Backend**:
   ```bash
   cd backend
   npm install
   node app.js
   ```

3. **Setup Frontend**:
   ```bash
   cd frontend
   npm install
   npm start
   ```

4. **Setup AI Models**:
   ```bash
   cd ai-models
   pip install -r requirements.txt
   python facemodel.py
   ```

---

## Usage
1. Start the backend server:
   ```bash
   node backend/app.js
   ```
2. Start the React frontend:
   ```bash
   npm start
   ```
3. Interact with the Emotion-Based Entertainment Assistant from the web interface.

---

## Future Enhancements
- Enhance the recommendation engine using Reinforcement Learning.
- Add support for multilingual voice tone analysis.
- Expand integration with IoT devices for a seamless experience.

---

## Contributing
Feel free to contribute by opening an issue or creating a pull request. Follow the contribution guidelines outlined in `CONTRIBUTING.md`.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Authors
- **[Your Name]**
- **Contributors**: Add here if applicable.

