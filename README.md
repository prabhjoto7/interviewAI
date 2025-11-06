# AI Interview System - Modular Architecture

## 📁 Project Structure

```
ai_interview_system/
│
├── main_app.py              # Main integration file (run this)
├── recording_system.py      # Module 1: Recording & Violation Detection
├── analysis_system.py       # Module 2: Multi-Modal Analysis
├── scoring_dashboard.py     # Module 3: Scoring & Dashboard
└── README.md               # This file
```

## 🎯 Module Overview

### **Module 1: `recording_system.py`**
**Real-Time Interview Recording and Violation Detection System**

**Responsibilities:**
- Video and audio recording
- Real-time violation detection (multiple people, looking away, no face, cheating items)
- Eye contact tracking
- Blink detection
- Head pose estimation
- Lighting analysis
- Audio transcription

**Key Class:** `RecordingSystem`

**Main Method:** `record_interview(question_data, duration, ui_callbacks)`

---

### **Module 2: `analysis_system.py`**
**Multi-Modal Analysis System**

**Responsibilities:**
- Facial emotion analysis (DeepFace)
- Audio quality assessment (fluency, accuracy, WPM)
- Visual outfit analysis (YOLO)
- Semantic similarity scoring
- Emotion aggregation and fusion

**Key Class:** `AnalysisSystem`

**Main Method:** `analyze_recording(recording_data, question_data, duration)`

---

### **Module 3: `scoring_dashboard.py`**
**Scoring, Hiring Decision, and Results Dashboard**

**Responsibilities:**
- Calculate hiring decision based on metrics
- Display immediate question results
- Render performance overview dashboard
- Question-by-question detailed analysis
- CSV export functionality

**Key Class:** `ScoringDashboard`

**Main Methods:** 
- `decide_hire(result)`
- `render_dashboard(results)`

---

### **Integration File: `main_app.py`**
**Main Application Entry Point**

**Responsibilities:**
- Load all AI models (once, with caching)
- Initialize all three systems
- Handle Streamlit UI and routing
- Manage session state
- Coordinate data flow between modules

---

## 🔌 How Modules Communicate

### **Loose Coupling Design**

Each module is **completely independent** and communicates through **standardized dictionaries**:

```python
# Module 1 Output → Module 2 Input
recording_data = {
    'video_path': str,
    'audio_path': str,
    'frames': list,
    'transcript': str,
    'eye_contact_pct': float,
    'blink_count': int,
    'face_box': tuple,
    'violation_detected': bool,
    'violation_reason': str,
    'violations': list
}

# Module 2 Output → Module 3 Input
analysis_results = {
    'fused_emotions': dict,
    'emotion_scores': dict,
    'accuracy': float,
    'fluency': float,
    'wpm': float,
    'outfit': str,
    'has_valid_data': bool
}

# Module 3 Output
final_result = {
    'hire_decision': str,
    'hire_reasons': list,
    ... (all previous data merged)
}
```

---

## ✅ Benefits of This Architecture

### **1. Independent Development**
- Modify `recording_system.py` without touching analysis logic
- Update `analysis_system.py` algorithms without affecting UI
- Change `scoring_dashboard.py` visualizations without breaking recording

### **2. Easy Testing**
```python
# Test Module 1 independently
recording_system = RecordingSystem(models)
result = recording_system.record_interview(question, 20, callbacks)

# Test Module 2 independently
analysis_system = AnalysisSystem(models)
analysis = analysis_system.analyze_recording(recording_data, question)

# Test Module 3 independently
dashboard = ScoringDashboard()
decision, reasons = dashboard.decide_hire(merged_result)
```

### **3. Easy Extension**
Want to add a new feature? Just modify one module:

- **New violation rule** → Edit `recording_system.py`
- **New emotion detection** → Edit `analysis_system.py`
- **New chart/metric** → Edit `scoring_dashboard.py`

### **4. Reusability**
Each module can be imported and used in other projects:

```python
# Use only the recording system in another app
from recording_system import RecordingSystem
recorder = RecordingSystem(models)
```

---

## 🚀 How to Run

### **1. Install Dependencies**
```bash
pip install streamlit opencv-python numpy pandas deepface mediapipe ultralytics sentence-transformers speechrecognition pyaudio
```

### **2. Run the Application**
```bash
streamlit run main_app.py
```

### **3. Project Structure**
Make sure all 4 files are in the same directory:
```
your_folder/
├── main_app.py
├── recording_system.py
├── analysis_system.py
└── scoring_dashboard.py
```

---

## 🔧 Customization Guide

### **Change Violation Rules**
Edit `recording_system.py`:
```python
# In record_interview() method, adjust thresholds:
if elapsed > 3.0:  # Change from 2.0 to 3.0 seconds
    self.violation_detected = True
```

### **Change Analysis Algorithms**
Edit `analysis_system.py`:
```python
# In evaluate_english_fluency(), adjust weights:
combined = (0.4 * alpha_ratio) + (0.3 * len_score) + ...
```

### **Change Scoring Logic**
Edit `scoring_dashboard.py`:
```python
# In decide_hire(), adjust thresholds:
if pos >= 6:  # More strict (was 5)
    decision = "✅ Hire"
```

### **Change UI/Dashboard**
Edit `scoring_dashboard.py` or `main_app.py`:
```python
# Add new charts, change colors, modify layout
```

---

## 🎨 Module Interfaces (API)

### **RecordingSystem API**
```python
class RecordingSystem:
    def __init__(self, models_dict)
    def record_interview(self, question_data, duration, ui_callbacks) -> dict
    def detect_cheating_items(self, detected_objects) -> list
    def calculate_eye_gaze(self, face_landmarks, frame_shape) -> bool
    def estimate_head_pose(self, face_landmarks, frame_shape) -> tuple
```

### **AnalysisSystem API**
```python
class AnalysisSystem:
    def __init__(self, models_dict)
    def analyze_recording(self, recording_data, question_data, duration) -> dict
    def analyze_frame_emotion(self, frame_bgr) -> dict
    def evaluate_answer_accuracy(self, answer, question, ideal) -> float
    def evaluate_english_fluency(self, text) -> float
    def analyze_outfit(self, frame, face_box) -> tuple
```

### **ScoringDashboard API**
```python
class ScoringDashboard:
    def __init__(self)
    def decide_hire(self, result) -> tuple
    def render_dashboard(self, results) -> None
    def display_immediate_results(self, result) -> None
    def export_results_csv(self, results) -> str
```

---

## 📦 Dependencies by Module

### **Module 1 (recording_system.py)**
- cv2 (opencv-python)
- numpy
- mediapipe
- ultralytics
- speech_recognition

### **Module 2 (analysis_system.py)**
- cv2 (opencv-python)
- numpy
- pandas
- deepface
- sentence-transformers
- ultralytics

### **Module 3 (scoring_dashboard.py)**
- streamlit
- numpy
- pandas

### **Main App (main_app.py)**
- streamlit
- All dependencies from modules 1-3

---

## 🛡️ Error Handling

Each module handles its own errors:

- **Module 1**: Returns `{'error': 'message'}` if camera fails
- **Module 2**: Returns default values (0.0) if analysis fails
- **Module 3**: Handles missing data gracefully in UI

The main app checks for errors and displays appropriate messages.

---

## 🔄 Data Flow Diagram

```
┌─────────────────┐
│   main_app.py   │
│  (Orchestrator) │
└────────┬────────┘
         │
         ├──► 1. Load Models (cached)
         │
         ├──► 2. RecordingSystem.record_interview()
         │         │
         │         └──► Returns: recording_data
         │
         ├──► 3. AnalysisSystem.analyze_recording(recording_data)
         │         │
         │         └──► Returns: analysis_results
         │
         ├──► 4. Merge recording_data + analysis_results
         │
         └──► 5. ScoringDashboard.decide_hire(merged_result)
                   │
                   └──► Returns: (decision, reasons)
```

---

## 💡 Best Practices

1. **Never modify dictionary keys** between modules - this breaks compatibility
2. **Always provide default values** in case of missing data
3. **Use type hints** when adding new methods
4. **Test each module independently** before integration
5. **Keep UI logic in main_app.py** or scoring_dashboard.py only

---

## 📝 Version History

- **v2.0**: Modular architecture with 3 independent systems
- **v1.0**: Monolithic single-file application

---

## 🤝 Contributing

When adding features:

1. Identify which module it belongs to
2. Add method to that module only
3. Update the module's docstrings
4. Test independently before integration
5. Update this README if adding new APIs

---

## 📧 Support

For questions about:
- **Recording issues** → Check `recording_system.py`
- **Analysis issues** → Check `analysis_system.py`
- **UI/Dashboard issues** → Check `scoring_dashboard.py` or `main_app.py`

---

**Built with ❤️ using Modular Design Principles**