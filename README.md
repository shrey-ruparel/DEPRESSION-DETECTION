# 🧠 Multimodal Depression Detection System

 A research-oriented system that fuses facial, vocal, and behavioral signals to assess emotional well-being and surface early indicators of depression.

---

## 📌 Overview

Traditional mental health screening tools rely on self-reported questionnaires alone — a single modality prone to bias and inaccuracy. This project takes a different approach.

The **Multimodal Depression Detection System** combines three independent input streams — **quiz responses, facial expressions, and voice signals** — into a unified risk assessment pipeline. Each modality provides complementary information; fusing them yields a more robust and reliable output than any single source could provide.

The system is designed for educational and research use, not clinical deployment.

---

## 🎯 Objectives

- Detect early signs of depression using multiple, independent data sources
- Improve prediction reliability through multimodal fusion
- Output a structured **risk classification** — Low, Medium, or High
- Deliver **actionable, personalized recommendations** based on assessed risk
- Track mood trends over time for longitudinal insight

---

## ⚙️ Features

### 1. Quiz-Based Psychological Assessment
- User completes a structured questionnaire based on validated psychological indicators
- Responses are classified using a **Naive Bayes model** trained on labeled data
- Outputs a normalized quiz score feeding into the fusion layer

### 2. Facial Emotion Detection
- Real-time webcam capture via **OpenCV**
- Emotion classification using a pretrained **TensorFlow/Keras** model
- Detects states: Happy, Sad, Neutral, Angry, Fearful, Surprised, Disgusted
- Outputs dominant emotion and confidence score

### 3. Voice Emotion Analysis
- Records live audio using **Sounddevice**
- Extracts acoustic features via **Librosa**:
  - **Energy** — overall speech loudness and vitality
  - **Pitch (F0)** — fundamental frequency contour
  - **MFCCs** — mel-frequency cepstral coefficients capturing vocal timbre
- Maps feature patterns to depressive vs. non-depressive speech profiles

### 4. Multimodal Fusion Engine
- Aggregates normalized scores from all three modules
- Applies weighted fusion logic to compute a **final composite score**
- Designed to be modular — individual components can be updated independently

### 5. Risk Classification
| Score Range | Risk Level  |
|:-----------:|:-----------:|
| 0 – 2       | 🟢 Low      |
| 3 – 5       | 🟡 Medium   |
| 6+          | 🔴 High     |

### 6. Personalized Recommendation Engine
Risk-stratified output delivers targeted suggestions:
- **Low**: Mindfulness exercises, sleep hygiene tips
- **Medium**: Journaling prompts, stress management techniques
- **High**: Professional consultation, crisis resources

### 7. Mood Tracking *(Optional)*
- Logs assessment history per user session
- Visualizes emotional trends over time (improving / stable / worsening)

---

## 🛠️ Tech Stack

| Layer              | Technology                          |
|--------------------|-------------------------------------|
| **Frontend**       | HTML, CSS, JavaScript               |
| **Backend**        | Flask (Python)                      |
| **ML — Quiz**      | Scikit-learn (Naive Bayes)          |
| **ML — Face**      | OpenCV, TensorFlow / Keras          |
| **ML — Voice**     | Librosa, Sounddevice                |
| **Database**       | SQLite / MySQL                      |

---

## 📂 Project Structure

```
DEPRESSION-DETECTION/
│
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── script.js
│
├── backend/
│   ├── app.py
│   └── modules/
│       ├── quiz.py
│       ├── face.py
│       ├── voice.py
│       └── fusion.py
│
├── models/
│   └── quiz_model.pkl
│
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Conda (recommended) or virtualenv
- Webcam and microphone access

### 1. Clone the Repository
```bash
git clone https://github.com/ARYAN-3345/DEPRESSION-DETECTION.git
cd DEPRESSION-DETECTION
```

### 2. Create and Activate Environment
```bash
conda create -n depression_env python=3.10
conda activate depression_env
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python backend/app.py
```

### 5. Open in Browser
```
http://127.0.0.1:5000/
```

---

## 🧪 System Pipeline

```
User Input
    │
    ├──── Quiz Responses  ──────► Naive Bayes Model  ──► Score_Q
    │
    ├──── Webcam Feed     ──────► CNN (Face Model)   ──► Score_F
    │
    └──── Microphone      ──────► Feature Extraction ──► Score_V
                                                              │
                                                    ┌─────────▼──────────┐
                                                    │   Fusion Engine    │
                                                    │ Final = Q + F + V  │
                                                    └─────────┬──────────┘
                                                              │
                                                    Risk Level + Recommendations
```

---

## ⚠️ Limitations

- **Noise sensitivity**: Voice module performance degrades in high-ambient-noise environments
- **Facial occlusion**: Masks, glasses, or poor lighting reduce facial detection accuracy
- **Cultural bias**: Emotion expression norms vary across demographics; model may not generalize universally
- **Single-session scope**: A single assessment captures a moment, not a pattern — longitudinal use increases reliability
- **Not a diagnostic tool**: This system does not meet clinical standards and cannot replace professional psychiatric evaluation

---

## 👨‍💻 Team

| Member       | Responsibility                        |
|--------------|---------------------------------------|
| **Shrey**    | ML Model Development & Fusion Logic   |
| **Rupashri** | Frontend UI & Backend Integration     |
| **Rutuja**   | Facial Emotion Detection Module       |
| **Aryan**    | Voice Analysis Module                 |

---

## 🔭 Future Work

- Continuous real-time monitoring (streaming inference)
- Deep learning–based voice emotion classification (e.g., wav2vec 2.0)
- Mobile application (React Native / Flutter)
- Improved personalization using longitudinal user history
- Multilingual support for broader accessibility

---

## 📜 Disclaimer

This project is developed strictly for **academic and educational purposes**. It is **not a certified medical device** and should **not** be used as a substitute for professional psychological assessment or clinical diagnosis. If you or someone you know is experiencing mental health difficulties, please consult a qualified healthcare professional.

---

## 🙏 Acknowledgements

- Publicly available depression and emotion datasets used for model training
- Python open-source ecosystem: Scikit-learn, TensorFlow, OpenCV, Librosa, Flask
- Academic literature on affective computing and multimodal sentiment analysis
