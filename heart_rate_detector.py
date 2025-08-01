import cv2
import numpy as np
from vitallens import VitalLens, Method, Mode
import time
from collections import deque
import logging
import tempfile
import os
import builtins

class HeartRateDetector:
    def __init__(self):
        self.vl = None
        self.cap = None
        self.is_running = False
        self.fps = 30
        self.buffer_size = 450  # 15 seconds
        self.min_confidence = 0.6
        self.heart_rate_buffer = deque(maxlen=300)
        self.last_analysis = time.time()
        self.analysis_interval = 1.0
        
        # Suppress logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Suppress VitalLens logging"""
        logging.getLogger('vitallens').setLevel(logging.CRITICAL + 1)
        logging.getLogger().handlers = []
        
        # Redirect output to temp directory
        temp_dir = tempfile.gettempdir()
        os.environ['VITALLENS_OUTPUT_DIR'] = temp_dir
        os.environ['VITALLENS_NO_LOG'] = '1'
        
        # Monkey-patch file opening to block JSON writes
        original_open = builtins.open
        def patched_open(*args, **kwargs):
            if args and isinstance(args[0], str) and args[0].endswith('.json'):
                return open(os.devnull, 'w')
            return original_open(*args, **kwargs)
        builtins.open = patched_open
        
    def initialize_camera(self):
        """Initialize webcam"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Cannot open webcam")
            
        # Camera settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
    def initialize_vitallens(self):
        """Initialize VitalLens for heart rate detection"""
        self.vl = VitalLens(
            method=Method.G,
            mode=Mode.BURST,
            detect_faces=True
        )
        
    def process_frame(self, frame):
        """Process a single frame for heart rate detection"""
        # Add frame to buffer
        self.heart_rate_buffer.append(frame.copy())
        
        # Keep only recent frames
        if len(self.heart_rate_buffer) > 300:
            self.heart_rate_buffer.popleft()
            
        # Analyze periodically
        current_time = time.time()
        if len(self.heart_rate_buffer) >= 150 and (current_time - self.last_analysis) >= self.analysis_interval:
            try:
                # Convert to numpy array
                video_array = np.array(list(self.heart_rate_buffer))
                
                # Analyze with VitalLens
                result = self.vl(video_array, fps=self.fps)
                
                if result and len(result) > 0:
                    face_data = result[0]
                    if 'vital_signs' in face_data and 'heart_rate' in face_data['vital_signs']:
                        hr_data = face_data['vital_signs']['heart_rate']
                        if hr_data['value'] is not None and hr_data.get('confidence', 0) >= self.min_confidence:
                            heart_rate = hr_data['value']
                            confidence = hr_data.get('confidence', 0)
                            
                            self.last_analysis = current_time
                            return heart_rate, confidence
                            
            except Exception as e:
                print(f"Analysis error: {e}")
                
        return None, 0.0
        
    def get_heart_rate_data(self, duration=30):
        """Get heart rate data for specified duration"""
        print(f"🔄 Collecting heart rate data for {duration} seconds...")
        print("Please stay still and look at the camera")
        
        heart_rates = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            heart_rate, confidence = self.process_frame(frame)
            if heart_rate and confidence >= self.min_confidence:
                heart_rates.append(heart_rate)
                
            # Display frame
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Collecting: {len(heart_rates)} samples", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Time: {duration - int(time.time() - start_time)}s", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            if heart_rate:
                cv2.putText(display_frame, f"HR: {heart_rate:.1f} BPM", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            cv2.imshow('Heart Rate Collection', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cv2.destroyAllWindows()
        
        if len(heart_rates) >= 10:
            print(f"✅ Collected {len(heart_rates)} heart rate samples")
            return heart_rates
        else:
            print("❌ Insufficient heart rate data collected")
            return None
            
    def cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Restore original file open function
        builtins.open = builtins.__dict__.get('open', open)

def test_heart_rate_detection():
    """Test heart rate detection"""
    detector = HeartRateDetector()
    
    try:
        detector.initialize_camera()
        detector.initialize_vitallens()
        
        print("🫀 Heart Rate Detection Test")
        print("=" * 30)
        print("Instructions:")
        print("• Keep your face visible in the camera")
        print("• Use good, even lighting")
        print("• Stay as still as possible")
        print("• Press 'q' to quit")
        print()
        
        heart_rates = detector.get_heart_rate_data(30)
        
        if heart_rates:
            print(f"Average Heart Rate: {np.mean(heart_rates):.1f} BPM")
            print(f"Min Heart Rate: {np.min(heart_rates):.1f} BPM")
            print(f"Max Heart Rate: {np.max(heart_rates):.1f} BPM")
            print(f"Standard Deviation: {np.std(heart_rates):.1f} BPM")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        detector.cleanup()

if __name__ == "__main__":
    test_heart_rate_detection() 