# 🧠 AI-Based Emotion Detection System

A complete, production-grade, and professional **Facial Emotion Recognition (FER)** system built using **Python 3.11**, **OpenCV**, **PyTorch / Deep Learning**, and **Streamlit**.

The application accurately detects human faces from real-time webcam video feeds or uploaded image files and classifies their emotional state across seven fundamental human emotions.

---

## 🌟 Key Features

### 1. 🎥 Live Webcam Detection (Dual Modes)
- **Mode A: Real-Time Continuous Video Stream (`OpenCV VideoCapture`)**:
  - Live video capture with interactive **Start** and **Stop** controls.
  - Real-time face localization with modern corner brackets and color-coded emotion pills.
  - Dynamic FPS counter and live multi-face tracking.
  - Real-time confidence percentage and emotion probability meters.
- **Mode B: Browser Snapshot Camera (`st.camera_input`)**:
  - Native HTML5 in-browser camera capture compatible with all modern browsers without hardware device locks.
  - Instantaneous freeze-frame multi-face emotion breakdown.

### 2. 🖼️ Image Upload & Multi-Face Detection
- Supports **JPG**, **JPEG**, and **PNG** formats.
- Pre-loaded sample face portraits for 1-click instant testing.
- Capable of simultaneously detecting and analyzing **multiple faces in group pictures**.
- High-resolution cropped face thumbnails, bounding box coordinates, and dominant emotion prediction.
- Detailed horizontal probability bar chart comparing confidence across all 7 emotions.
- Graceful handling and informative alert banner when no face is found in an image.

### 3. 📊 Emotion Analytics Dashboard
- Session performance metric cards:
  - Total Faces Detected
  - Dominant / Most Frequent Emotion
  - Average Detection Confidence (%)
  - Peak Confidence Score (%)
- Interactive **Plotly Donut Chart** displaying the session-wide emotion distribution.
- Interactive **Plotly Column Bar Chart** tracking emotion frequency counts.
- Comprehensive session history log table with face thumbnails, emotion labels, timestamps, and detection source.
- **CSV Report Export**: Download full detection logs with timestamp and individual emotion percentages with one click.
- **Clear Session History**: Reset session analytics on demand.

### 4. 🎨 Modern Dashboard User Interface
- **Dark & Light Theme Toggle**: Seamless CSS variables delivering polished aesthetics in both modes.
- Modern glassmorphic cards, responsive columns, and subtle hover animations.
- Color-coded badges for every emotion.
- Sidebar navigation with system status indicators.

### 5. ℹ️ System Architecture & Theory
- Educational deep-dive explaining the **Mini-Xception Residual CNN** architecture.
- Historical background on the **FER-2013 dataset** (35,887 images).
- Step-by-step visual walkthrough of the computer vision pipeline.

---

## 🎭 Recognized Emotion Classes

| Emotion | Icon | Dominant Visual Facial Landmarks | Badge Color |
|---|:---:|---|:---:|
| **Happy** | 😄 | Smiling lips, raised cheeks, crow's feet at eye corners | `#10b981` |
| **Sad** | 😢 | Drooping upper eyelids, downward mouth corners, raised inner eyebrows | `#3b82f6` |
| **Angry** | 😠 | Eyebrows pulled down & together, glaring eyes, tightened lips | `#ef4444` |
| **Fear** | 😨 | Raised upper eyelids, eyebrows drawn up and together, parted lips | `#a855f7` |
| **Surprise** | 😲 | Eyebrows raised and curved, wide open eyes, dropped jaw | `#f59e0b` |
| **Neutral** | 😐 | Relaxed facial musculature, natural resting expression | `#94a3b8` |
| **Disgust** | 🤢 | Wrinkled nose bridge, raised upper lip, narrowed eyes | `#22c55e` |

---

## 🏗️ Project Architecture

```
AI_Emotion_Detection/
├── app.py                     # Streamlit multi-page dashboard application
├── emotion_detector.py        # Core Computer Vision & Deep Learning Engine
├── requirements.txt           # Python dependency specifications
├── README.md                  # Project documentation & execution guide
├── assets/
│   ├── style.css              # Custom responsive CSS and themes
│   └── sample_images/         # Test portrait faces (Happy, Neutral, etc.)
│       ├── happy_smile.jpg
│       ├── neutral_man.jpg
│       ├── sample_face.jpg
│       └── surprised_person.jpg
└── models/
    ├── emotion_model.pt       # PyTorch weights for Mini-Xception CNN
    ├── emotion_model.hdf5     # Pre-trained Keras model checkpoint
    └── haarcascade_frontalface_default.xml # OpenCV Haar Cascade classifier
```

---

## 🔬 How the Emotion Detection Pipeline Works

```
┌─────────────────┐     ┌─────────────────────┐     ┌───────────────────┐
│ Input Image /   │ ──> │ Grayscale & Hist.   │ ──> │ Haar Cascade Face │
│ Video Frame     │     │ Equalization        │     │ Localization      │
└─────────────────┘     └─────────────────────┘     └───────────────────┘
                                                              │
                                                              ▼
┌─────────────────┐     ┌─────────────────────┐     ┌───────────────────┐
│ Softmax Emotion │ <── │ Mini-Xception CNN   │ <── │ 64x64 Resize &    │
│ Probabilities   │     │ (Residual Blocks)   │     │ [-1, 1] Norm.     │
└─────────────────┘     └─────────────────────┘     └───────────────────┘
```

1. **Face Localization**: OpenCV's Haar Cascade detects facial bounding boxes `(x, y, w, h)`.
2. **Padding & Aspect Ratio**: A 10% safety margin is applied, and the crop is converted to a square to prevent spatial distortion.
3. **Preprocessing**: Grayscale conversion, resizing to `64 × 64` pixels, and floating point normalization to `[-1.0, 1.0]`.
4. **Mini-Xception CNN**: Employs Depthwise Separable Convolutions and Residual identity mappings to extract high-level emotion features while keeping latency under 5 milliseconds.
5. **Softmax Output**: Generates probability scores summing to 100% across the 7 emotion categories.

---

## 💻 Installation & Execution Instructions (Windows & VS Code)

### Prerequisites
- Windows 10 or 11
- Python 3.11 (64-bit) installed and added to `PATH`
- Visual Studio Code (recommended) or any standard terminal

---

### Step 1: Open Terminal in Project Directory
Open **VS Code**, then open the project folder:
```powershell
cd c:\Users\tejan\Downloads\AI_EMOTION_DETECTION
```

---

### Step 2: Create a Virtual Environment (Recommended)
Create an isolated Python 3.11 virtual environment named `venv`:
```powershell
python -m venv venv
```

---

### Step 3: Activate the Virtual Environment
In Windows PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
```
*(If PowerShell blocks script execution, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once).*

Or in standard Windows Command Prompt (cmd.exe):
```cmd
venv\Scripts\activate.bat
```

---

### Step 4: Upgrade Pip & Install Dependencies
Install all required packages from `requirements.txt`:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 5: Run the Streamlit Application
Launch the interactive web application:
```powershell
streamlit run app.py
```

The application will automatically open in your default browser at:
👉 **`http://localhost:8501`**

---

## 🛠️ Troubleshooting & Tips

### Webcam Not Detected or Permission Denied
- If another application (Zoom, Teams, Skype, or Windows Camera) is using the webcam, close it before starting the camera in the app.
- Check Windows Camera Privacy Settings: **Settings > Privacy & security > Camera > Allow desktop apps to access your camera**.
- If your system has multiple cameras (e.g. built-in laptop camera vs external USB webcam), select **Camera 1** or **Camera 2** from the device dropdown.
- Alternatively, use the **Browser Snapshot Camera** tab, which uses browser-managed permissions.

### Testing Without a Physical Webcam
- Switch to the **Image Upload & Analysis** page.
- Choose any of the pre-loaded sample faces from the dropdown (e.g., `Happy Smile`, `Neutral Man`, `Surprised Person`) to test the model immediately.

---

## 📜 Technology Stack

- **Computer Vision:** OpenCV (`cv2`)
- **Deep Learning:** PyTorch (`torch`, `torch.nn`), FER
- **Frontend Dashboard:** Streamlit
- **Data Visualizations:** Plotly (`plotly.graph_objects`)
- **Image Processing:** Pillow (`PIL`), NumPy
- **Exporting:** Python standard `csv` module

---

## 📄 License
This project is developed for educational and professional demonstration purposes. Model weights and Haar cascade files are licensed under the MIT License.
