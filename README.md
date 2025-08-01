# 🫀 Heart Rate Lock System

A personalized lock system that uses heart rate patterns from rPPG (remote Photoplethysmography) for authentication. This innovative system leverages the unique heart rate patterns of individuals as a biometric identifier.

## 🌟 Features

- **One-Shot Learning**: Create user profiles with just 30 seconds of heart rate data
- **Real-Time Detection**: Continuous heart rate monitoring using webcam
- **Pattern Analysis**: Advanced heart rate pattern recognition using statistical and frequency domain features
- **Modern GUI**: Beautiful and intuitive user interface with real-time visualization
- **Secure Authentication**: Cosine similarity-based pattern matching with configurable thresholds
- **Multi-User Support**: Support for multiple user profiles
- **Visualization**: Real-time heart rate pattern plotting and comparison tools

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Webcam
- Good lighting conditions

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd heart-rate-lock-system
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the main application:**
   ```bash
   python heart_rate_lock.py
   ```

## 📁 Project Structure

```
heart-rate-lock-system/
├── heart_rate_lock.py          # Main lock system with GUI
├── heart_rate_detector.py      # Standalone heart rate detection
├── pattern_analyzer.py         # Heart rate pattern analysis
├── gui_interface.py           # GUI interface module
├── simple_realtime_hr.py      # Original simple heart rate detection
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 🔧 Usage

### 1. Main Lock System (`heart_rate_lock.py`)

The complete heart rate lock system with GUI:

```bash
python heart_rate_lock.py
```

**Features:**
- Calibration mode for creating user profiles
- Real-time heart rate monitoring
- Pattern-based authentication
- Visual heart rate pattern display
- System lock/unlock functionality

### 2. Heart Rate Detector (`heart_rate_detector.py`)

Standalone heart rate detection for testing:

```bash
python heart_rate_detector.py
```

**Features:**
- Simple heart rate detection
- Data collection for analysis
- Basic visualization

### 3. Pattern Analyzer (`pattern_analyzer.py`)

Advanced heart rate pattern analysis:

```bash
python pattern_analyzer.py
```

**Features:**
- Statistical feature extraction
- FFT-based frequency analysis
- Pattern similarity calculation
- Visualization tools

### 4. GUI Interface (`gui_interface.py`)

Modern GUI for the lock system:

```bash
python gui_interface.py
```

**Features:**
- Real-time camera feed
- Heart rate monitoring
- Pattern visualization
- System controls

### 5. Simple Real-Time HR (`simple_realtime_hr.py`)

Original simple heart rate detection:

```bash
python simple_realtime_hr.py
```

**Features:**
- Basic real-time heart rate detection
- OpenCV overlay display
- Console output

## 🎯 How It Works

### 1. Heart Rate Detection
- Uses **VitalLens** library for rPPG (remote Photoplethysmography)
- Captures video frames from webcam
- Extracts heart rate signals from facial blood flow
- Processes signals in real-time

### 2. Pattern Analysis
- **Statistical Features**: Mean, standard deviation, skewness, kurtosis
- **Peak Analysis**: Counts peaks, valleys, and zero crossings
- **Frequency Domain**: FFT-based feature extraction
- **Autocorrelation**: Signal correlation analysis

### 3. One-Shot Learning
- Creates user profile from 30 seconds of heart rate data
- Extracts comprehensive pattern features
- Stores profile for future authentication

### 4. Authentication
- Compares current heart rate pattern with stored profile
- Uses cosine similarity for pattern matching
- Configurable similarity threshold (default: 0.85)
- Real-time authentication decision

## ⚙️ Configuration

### Similarity Threshold
Adjust the similarity threshold in the GUI or code:
```python
similarity_threshold = 0.85  # Higher = more strict
```

### Confidence Threshold
Set minimum confidence for heart rate readings:
```python
min_confidence = 0.6  # Lower = more readings, less accurate
```

### Analysis Parameters
- **Buffer Size**: 450 frames (15 seconds at 30 FPS)
- **Analysis Interval**: 3 seconds
- **Pattern Length**: 100 points for feature extraction

## 🔒 Security Features

### Pattern-Based Authentication
- Uses heart rate **patterns**, not absolute values
- Resistant to heart rate variations due to exercise, stress, etc.
- Unique to each individual's cardiovascular characteristics

### One-Shot Learning
- Requires only one training session
- No need for multiple samples
- Quick profile creation

### Real-Time Monitoring
- Continuous authentication
- Immediate lock/unlock response
- Visual feedback

## 📊 Visualization

### Real-Time Plots
- Current heart rate pattern
- User profile comparison
- Similarity score display
- Confidence indicators

### Pattern Analysis
- Raw vs. smoothed heart rate data
- Statistical feature visualization
- Pattern comparison tools

## 🛠️ Technical Details

### Dependencies
- **OpenCV**: Video capture and processing
- **VitalLens**: rPPG heart rate detection
- **NumPy**: Numerical computations
- **SciPy**: Signal processing
- **scikit-learn**: Machine learning utilities
- **Matplotlib**: Visualization
- **Tkinter**: GUI framework

### Algorithm
1. **Signal Acquisition**: Capture video frames
2. **rPPG Processing**: Extract heart rate from facial blood flow
3. **Feature Extraction**: Calculate statistical and frequency features
4. **Pattern Matching**: Compare with stored user profile
5. **Authentication**: Make lock/unlock decision

### Performance
- **Latency**: ~3 seconds for analysis
- **Accuracy**: Depends on lighting and user stillness
- **Memory**: ~15MB for video buffer
- **CPU**: Moderate usage during analysis

## 🎨 GUI Features

### Main Interface
- **Status Display**: System lock/unlock status
- **Heart Rate Monitor**: Real-time BPM display
- **Confidence Indicator**: Signal quality feedback
- **Similarity Score**: Pattern match percentage

### Controls
- **Calibrate**: Create new user profile
- **Monitor**: Start/stop heart rate monitoring
- **Lock/Unlock**: Manual system control
- **Settings**: Adjust thresholds and parameters

### Visualization
- **Real-time Plot**: Current heart rate pattern
- **Camera Feed**: Live video with overlay
- **System Log**: Activity and error messages

## 🔧 Troubleshooting

### Common Issues

1. **"Cannot open webcam"**
   - Check webcam permissions
   - Ensure no other application is using the camera
   - Try different camera index (0, 1, 2...)

2. **Low confidence readings**
   - Improve lighting conditions
   - Stay still during calibration
   - Ensure face is clearly visible

3. **Poor authentication accuracy**
   - Recalibrate with better conditions
   - Adjust similarity threshold
   - Check for environmental factors

4. **High CPU usage**
   - Reduce analysis frequency
   - Lower video resolution
   - Close other applications

### Performance Tips

- **Lighting**: Use even, bright lighting
- **Position**: Stay 30-60cm from camera
- **Stillness**: Minimize movement during calibration
- **Environment**: Avoid fluorescent lighting
- **Hardware**: Use USB 3.0 webcam for better performance

## 🚀 Advanced Usage

### Custom Pattern Analysis
```python
from pattern_analyzer import HeartRatePatternAnalyzer

analyzer = HeartRatePatternAnalyzer()
pattern = analyzer.extract_pattern_features(heart_rates)
similarity = analyzer.calculate_similarity(pattern1, pattern2)
```

### Integration with Other Systems
```python
from heart_rate_lock import HeartRateLock

lock_system = HeartRateLock()
if lock_system.authenticate_user(current_pattern):
    # Unlock system
    pass
```

### Batch Processing
```python
from heart_rate_detector import HeartRateDetector

detector = HeartRateDetector()
heart_rates = detector.get_heart_rate_data(duration=60)
```

## 📈 Future Enhancements

- **Multi-modal Authentication**: Combine with facial recognition
- **Cloud Profiles**: Store profiles securely in cloud
- **Mobile App**: Companion mobile application
- **API Integration**: REST API for external systems
- **Advanced ML**: Deep learning for better pattern recognition
- **Biometric Fusion**: Combine multiple biometric modalities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **VitalLens**: For rPPG heart rate detection
- **OpenCV**: For computer vision capabilities
- **SciPy**: For signal processing tools
- **Matplotlib**: For visualization features

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact the development team
- Check the troubleshooting section

---

**🫀 Heart Rate Lock System** - Secure, innovative, and personalized authentication using your unique heart rate pattern. 