# ASL Translator

Real-time American Sign Language (ASL) to text translator using MediaPipe hand landmarks and a BiLSTM model. A webcam feed is processed frame-by-frame; the backend accumulates 30 frames into a landmark sequence, runs the model, and streams predictions to the browser UI.

---

## Project Structure

```
asl-translator/
├── backend/
│   ├── app.py            # FastAPI server (endpoints: /predict, /reset, /classes, /tts, /health)
│   ├── inference.py      # BiLSTM inference engine + MediaPipe landmark extraction
│   ├── tts_engine.py     # gTTS server-side text-to-speech (fallback)
│   └── test_backend.py   # Integration smoke tests
├── frontend/
│   ├── index.html        # Web UI
│   ├── app.js            # Webcam capture, API calls, UI updates
│   └── style.css         # Dark-theme styles
├── model/
│   ├── lstm_architecture.py   # BiLSTM model definition
│   ├── lstm_train.py          # Training script
│   ├── evaluate.py            # Evaluation (accuracy, F1, confusion matrix)
│   └── saved_lstm/            # Trained model weights + class labels
├── scripts/
│   ├── download_dataset.py    # Download raw MP4 clips from Kaggle
│   ├── preprocess_landmarks.py# Extract MediaPipe landmarks → .npy sequences
│   └── verify_data.py         # Validate processed dataset
├── demo/
│   └── realtime_demo.py       # Standalone OpenCV real-time demo
├── data/                      # Raw MP4 clips + processed landmark .npy files
├── requirements.txt
└── .env                       # Kaggle API credentials (not committed)
```

---

## Supported Signs (20 classes)

`all` `before` `black` `book` `candy` `chair` `clothes` `computer` `cousin` `deaf` `drink` `fine` `go` `help` `no` `thin` `walk` `who` `year` `yes`

---

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. (First time) Download dataset from Kaggle
#    Add your credentials to .env, then:
python scripts/download_dataset.py

# 4. Extract landmarks from raw video
python scripts/preprocess_landmarks.py

# 5. Train the BiLSTM model
python model/lstm_train.py

# 6. (Optional) Evaluate on test set
python model/evaluate.py
```

---

## Running the App

**Terminal 1 — Backend:**
```bash
uvicorn backend.app:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
# Any static file server, e.g.:
cd frontend
python -m http.server 3000
# Then open http://localhost:3000
```

Click **Start** in the browser, allow camera access, and begin signing.

---

## Running the OpenCV Demo (no browser needed)

```bash
python demo/realtime_demo.py
```

Keyboard controls: `Q` quit · `C` clear buffer · `S` toggle smoothing

---

## Running Tests

```bash
# Start the backend first, then:
pytest backend/test_backend.py -v

# Or without pytest:
python backend/test_backend.py
```

---

## Model

| Architecture | BiLSTM (Bidirectional LSTM) |
|---|---|
| Input | 30 frames × 63 landmarks (21 keypoints × xyz) |
| Classes | 20 ASL words |
| Val accuracy | 74.3 % |
| Test accuracy | 65.7 % |
| Top-5 accuracy | 88.6 % |

Landmarks are extracted with MediaPipe Hands, wrist-normalized, and scale-normalized before feeding to the model.

---

## Tech Stack

- **ML:** TensorFlow 2.16 / Keras 3.3, MediaPipe 0.10
- **Backend:** FastAPI, Uvicorn, gTTS
- **Frontend:** Vanilla JS, Web Speech API
- **Data:** OpenCV, Albumentations, scikit-learn
