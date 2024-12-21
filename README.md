# Release Notes for Emotion Recognition Project

## Version: 1.0.0

### Overview

This release introduces a Python-based emotion recognition system capable of detecting emotions from facial expressions using a pre-trained model and triggering mood-specific music playback. The project leverages the FER2013 dataset for training and uses a more robust model for enhanced emotion detection accuracy.

### Features

1. **Emotion Detection**:

   - Utilizes a pre-trained model for accurate emotion detection.
   - Detects emotions such as happiness, sadness, anger, surprise, and more.

2. **Mood Music Playback**:

   - Automatically plays a mood-specific music track corresponding to the detected emotion.

3. **Model Management**:

   - Includes an old version of the emotion recognition model (`my_model_old_version`) for historical reference.
   - The main application script (`main.py`) in the root directory uses the `face_model.weights.h5` model, which is better trained and provides more accurate emotion detection.
   - The `main.py` script in the root directory also supports mood-specific music playback after detecting an emotion.

4. **Extensible Architecture**:

   - Organized directory structure for ease of development and model training.

### Project Structure

```
.
├── face_model.weights.h5             # Pre-trained model weights (default)
├── main.py                           # Main application script
├── my_model_old_version              # Directory for the older model
│   ├── json
│   │   └── emotion_model.json        # Model configuration (old version)
│   ├── main.py                       # Application script for old model
│   ├── models
│   │   └── emotion_model.weights.h5  # Old model weights
│   └── train.py                      # Script for training the old model
├── README.md                         # Project documentation
└── requirement.txt                   # Python dependencies
```

### Installation and Usage

#### Prerequisites

- Python 3.12

#### Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/Astramat/Emotion-recognition.git
   cd Emotion-recognition
   ```

2. Install required Python dependencies:

   ```bash
   pip install -r requirement.txt
   ```

3. Run the application:

   ```bash
   python main.py
   ```

#### Usage

- The application detects the user's emotion through facial expression analysis.
- After detecting the emotion with the `face_model.weights.h5` pre-trained model, it plays a music track to match the detected emotion's mood.
- To experiment with the old model, navigate to `my_model_old_version/` and run its `main.py` script:
  ```bash
  cd my_model_old_version
  python main.py
  ```
  Note: The old model is only used for emotion detection and does not include music playback functionality.

### Known Issues

- The old model (`my_model_old_version`) has lower accuracy compared to the default pre-trained model.

### Acknowledgments

- FER2013 dataset: [https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)
- Community contributions to the pre-trained model.
