const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');

const app = express();
const port = 5000;

// Middleware
app.use(cors());
app.use(bodyParser.json());

// Map emotions to entertainment genres
const mapEmotionToGenre = (emotion) => {
    const recommendations = {
        'Happy': ['Comedy', 'Upbeat Music', 'Feel-good Movies'],
        'Sad': ['Drama', 'Sad Movies', 'Relaxing Music'],
        'Angry': ['Action Movies', 'Heavy Metal Music', 'Thrillers'],
        'Fear': ['Horror Movies', 'Intense Music'],
        'Neutral': ['Documentaries', 'Classical Music'],
        'Surprise': ['Mystery Movies', 'Exciting Music'],
        'Disgust': ['Dark Comedy', 'Alternative Music']
    };

    return recommendations[emotion] || ['No recommendations available'];
};

// Endpoint for recommendations based on emotion
app.post('/recommendations', (req, res) => {
    const { emotion } = req.body;

    if (!emotion) {
        return res.status(400).json({ error: 'Emotion not provided' });
    }

    const genres = mapEmotionToGenre(emotion);
    res.json({ recommendations: genres });
});

// Start the server
app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
});
