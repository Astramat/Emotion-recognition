### README for `my_model_old_version` Directory

---

## Overview

The `my_model_old_version` directory contains a previous version of the emotion recognition model. This model was initially used before the primary, more advanced model (`emotion_detection_model.h5`) was adopted.

### Features

1. **Emotion Detection**:
   - Trained using the FER2013 dataset.
   - Detects emotions such as happiness, sadness, anger, and surprise.
   - Lower accuracy compared to the newer model.

2. **Application**:
   - Primarily for emotion detection only, **without mood-based music playback**.

### Project Structure

```
my_model_old_version/
├── json
│   └── emotion_model.json        # Model configuration (old version)
├── main.py                       # Application script for the old model
├── models
│   └── emotion_model.weights.h5  # Old model weights
└── train.py                      # Script for training the old model
```

### Installation and Usage

#### Prerequisites

- Python 3.11

#### Installation Steps

1. Clone the repository if you haven't already:

   ```bash
   git clone https://github.com/Astramat/Emotion-recognition.git
   cd Emotion-recognition
   ```

2. Navigate to the `my_model_old_version` directory:

   ```bash
   cd my_model_old_version
   ```

3. Install required Python dependencies:

   ```bash
   pip install -r requirement.txt
   ```

4. Run the emotion detection application using the `main.py` script:

   ```bash
   python main.py
   ```

#### Usage

- The `main.py` script will detect emotions using the old model.
- It focuses solely on emotion recognition and does **not** include mood-based music playback.

---

### Known Issues

- The old model (`my_model_old_version`) has lower accuracy compared to the default pre-trained model (`emotion_detection_model.h5`).
- No mood-specific music playback functionality is available with this version.

---
