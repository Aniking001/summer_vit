import cv2
import numpy as np
from vitallens import VitalLens, Method, Mode
import time
from collections import deque
import os
import logging
import tempfile
import builtins
import json
import pickle
from datetime import datetime
import threading
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import savgol_filter
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation

class HeartRateLock:
    def __init__(self):
        self.vl = None
        self.cap = None
        self.is_running = False
        self.is_locked = True
        self.user_profile = None
        self.calibration_data = None
        self.heart_rate_buffer = deque(maxlen=300)  # 10 seconds at 30fps
        self.pattern_buffer = deque(maxlen=150)  # 5 seconds of processed data
        self.fps = 30
        self.min_confidence = 0.6
        self.similarity_threshold = 0.85
        self.last_analysis = time.time()
        self.analysis_interval = 1.0  # Analyze every second
        
        # GUI components
        self.root = None
        self.canvas = None
        self.status_label = None
        self.heart_rate_label = None
        self.pattern_plot = None
        
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
        
    def extract_heart_rate_pattern(self, heart_rates):
        """Extract heart rate pattern features"""
        if len(heart_rates) < 10:
            return None
            
        # Convert to numpy array
        hr_array = np.array(heart_rates)
        
        # Apply smoothing
        if len(hr_array) > 5:
            hr_smoothed = savgol_filter(hr_array, min(5, len(hr_array)//2), 2)
        else:
            hr_smoothed = hr_array
            
        # Extract features
        features = {
            'mean': np.mean(hr_smoothed),
            'std': np.std(hr_smoothed),
            'min': np.min(hr_smoothed),
            'max': np.max(hr_smoothed),
            'range': np.max(hr_smoothed) - np.min(hr_smoothed),
            'peaks': self._count_peaks(hr_smoothed),
            'pattern': hr_smoothed.tolist()[:50]  # First 50 points
        }
        
        return features
        
    def _count_peaks(self, signal):
        """Count peaks in the signal"""
        if len(signal) < 3:
            return 0
            
        peaks = 0
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                peaks += 1
        return peaks
        
    def calculate_similarity(self, pattern1, pattern2):
        """Calculate similarity between two heart rate patterns"""
        if not pattern1 or not pattern2:
            return 0.0
            
        # Compare pattern arrays
        if len(pattern1['pattern']) > 0 and len(pattern2['pattern']) > 0:
            # Pad shorter pattern
            max_len = max(len(pattern1['pattern']), len(pattern2['pattern']))
            p1 = pattern1['pattern'] + [pattern1['pattern'][-1]] * (max_len - len(pattern1['pattern']))
            p2 = pattern2['pattern'] + [pattern2['pattern'][-1]] * (max_len - len(pattern2['pattern']))
            
            # Calculate cosine similarity
            similarity = cosine_similarity([p1], [p2])[0][0]
            
            # Also compare statistical features
            stat_similarity = 1.0 - abs(pattern1['mean'] - pattern2['mean']) / max(pattern1['mean'], pattern2['mean'])
            stat_similarity = max(0, stat_similarity)  # Ensure non-negative
            
            # Combine similarities
            final_similarity = (similarity + stat_similarity) / 2
            return final_similarity
            
        return 0.0
        
    def authenticate_user(self, current_pattern):
        """Authenticate user using heart rate pattern"""
        if not self.user_profile or not current_pattern:
            return False
            
        similarity = self.calculate_similarity(self.user_profile, current_pattern)
        return similarity >= self.similarity_threshold
        
    def create_user_profile(self, heart_rates):
        """Create user profile from heart rate data"""
        pattern = self.extract_heart_rate_pattern(heart_rates)
        if pattern:
            self.user_profile = pattern
            self.save_profile()
            return True
        return False
        
    def save_profile(self):
        """Save user profile to file"""
        if self.user_profile:
            profile_data = {
                'pattern': self.user_profile,
                'created_at': datetime.now().isoformat(),
                'calibration_data': self.calibration_data
            }
            
            try:
                with open('user_profile.pkl', 'wb') as f:
                    pickle.dump(profile_data, f)
                print("✅ User profile saved")
            except Exception as e:
                print(f"❌ Error saving profile: {e}")
                
    def load_profile(self):
        """Load user profile from file"""
        try:
            if os.path.exists('user_profile.pkl'):
                with open('user_profile.pkl', 'rb') as f:
                    profile_data = pickle.load(f)
                    
                self.user_profile = profile_data['pattern']
                self.calibration_data = profile_data.get('calibration_data')
                print("✅ User profile loaded")
                return True
        except Exception as e:
            print(f"❌ Error loading profile: {e}")
        return False
        
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
                            
                            # Add to pattern buffer
                            self.pattern_buffer.append(heart_rate)
                            
                            # Extract pattern if we have enough data
                            if len(self.pattern_buffer) >= 50:
                                current_pattern = self.extract_heart_rate_pattern(list(self.pattern_buffer))
                                
                                if current_pattern:
                                    # Check authentication if we have a profile
                                    if self.user_profile and not self.is_locked:
                                        if self.authenticate_user(current_pattern):
                                            self.is_locked = False
                                        else:
                                            self.is_locked = True
                                            
                            return heart_rate, confidence
                            
            except Exception as e:
                print(f"Analysis error: {e}")
                
            self.last_analysis = current_time
            
        return None, 0.0
        
    def run_calibration(self):
        """Run calibration to create user profile"""
        print("🔄 Starting calibration...")
        print("Please stay still and look at the camera for 30 seconds")
        
        calibration_heart_rates = []
        start_time = time.time()
        
        while time.time() - start_time < 30:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            heart_rate, confidence = self.process_frame(frame)
            if heart_rate and confidence >= self.min_confidence:
                calibration_heart_rates.append(heart_rate)
                
            # Display frame
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Calibration: {len(calibration_heart_rates)} samples", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Time: {30 - int(time.time() - start_time)}s", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow('Calibration', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cv2.destroyAllWindows()
        
        if len(calibration_heart_rates) >= 20:
            if self.create_user_profile(calibration_heart_rates):
                print("✅ Calibration completed successfully!")
                return True
            else:
                print("❌ Calibration failed - insufficient data")
                return False
        else:
            print("❌ Calibration failed - need at least 20 heart rate samples")
            return False
            
    def create_gui(self):
        """Create the main GUI"""
        self.root = tk.Tk()
        self.root.title("Heart Rate Lock System")
        self.root.geometry("800x600")
        self.root.configure(bg='#2c3e50')
        
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', padding=10, font=('Arial', 10))
        style.configure('TLabel', font=('Arial', 12))
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="🫀 Heart Rate Lock System", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="System Status", padding="10")
        status_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        self.status_label = ttk.Label(status_frame, text="Initializing...", 
                                     font=('Arial', 12, 'bold'))
        self.status_label.grid(row=0, column=0, pady=5)
        
        self.heart_rate_label = ttk.Label(status_frame, text="Heart Rate: -- BPM", 
                                         font=('Arial', 10))
        self.heart_rate_label.grid(row=1, column=0, pady=5)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=20)
        
        self.calibrate_btn = ttk.Button(button_frame, text="🔧 Calibrate", 
                                       command=self.start_calibration)
        self.calibrate_btn.grid(row=0, column=0, padx=10)
        
        self.lock_btn = ttk.Button(button_frame, text="🔒 Lock", 
                                  command=self.lock_system)
        self.lock_btn.grid(row=0, column=1, padx=10)
        
        self.unlock_btn = ttk.Button(button_frame, text="🔓 Unlock", 
                                    command=self.unlock_system)
        self.unlock_btn.grid(row=0, column=2, padx=10)
        
        # Pattern visualization
        pattern_frame = ttk.LabelFrame(main_frame, text="Heart Rate Pattern", padding="10")
        pattern_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 20))
        
        # Create matplotlib figure for pattern visualization
        self.fig, self.ax = plt.subplots(figsize=(8, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, pattern_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Update button states
        self.update_button_states()
        
    def update_button_states(self):
        """Update button states based on system state"""
        if self.user_profile:
            self.calibrate_btn.config(text="🔄 Recalibrate")
        else:
            self.calibrate_btn.config(text="🔧 Calibrate")
            
        if self.is_locked:
            self.lock_btn.config(state='disabled')
            self.unlock_btn.config(state='normal')
        else:
            self.lock_btn.config(state='normal')
            self.unlock_btn.config(state='disabled')
            
    def start_calibration(self):
        """Start calibration process"""
        if messagebox.askyesno("Calibration", 
                              "This will create a new user profile.\n"
                              "Please stay still and look at the camera for 30 seconds.\n\n"
                              "Continue?"):
            self.root.withdraw()  # Hide main window
            success = self.run_calibration()
            self.root.deiconify()  # Show main window again
            
            if success:
                messagebox.showinfo("Success", "Calibration completed successfully!")
                self.update_button_states()
            else:
                messagebox.showerror("Error", "Calibration failed. Please try again.")
                
    def lock_system(self):
        """Lock the system"""
        self.is_locked = True
        self.update_button_states()
        self.update_status("🔒 System Locked")
        
    def unlock_system(self):
        """Unlock the system"""
        if self.user_profile:
            self.is_locked = False
            self.update_button_states()
            self.update_status("🔓 System Unlocked")
        else:
            messagebox.showwarning("Warning", "Please calibrate first!")
            
    def update_status(self, status):
        """Update status display"""
        if self.status_label:
            self.status_label.config(text=status)
            
    def update_heart_rate_display(self, heart_rate, confidence):
        """Update heart rate display"""
        if self.heart_rate_label:
            if heart_rate:
                self.heart_rate_label.config(
                    text=f"Heart Rate: {heart_rate:.1f} BPM (Confidence: {confidence:.2f})"
                )
            else:
                self.heart_rate_label.config(text="Heart Rate: -- BPM")
                
    def update_pattern_plot(self):
        """Update the pattern visualization"""
        if len(self.pattern_buffer) > 10:
            self.ax.clear()
            
            # Plot current pattern
            hr_data = list(self.pattern_buffer)
            self.ax.plot(hr_data, 'b-', label='Current Pattern', linewidth=2)
            
            # Plot user profile if available
            if self.user_profile and len(self.user_profile['pattern']) > 0:
                profile_data = self.user_profile['pattern']
                x_profile = np.linspace(0, len(hr_data)-1, len(profile_data))
                self.ax.plot(x_profile, profile_data, 'r--', label='User Profile', linewidth=2)
                
            self.ax.set_xlabel('Time')
            self.ax.set_ylabel('Heart Rate (BPM)')
            self.ax.set_title('Heart Rate Pattern')
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)
            
            self.canvas.draw()
            
    def run_gui(self):
        """Run the GUI main loop"""
        self.create_gui()
        
        # Load existing profile
        if self.load_profile():
            self.update_status("✅ Profile loaded - System Ready")
        else:
            self.update_status("⚠️ No profile found - Please calibrate")
            
        # Start camera thread
        self.is_running = True
        camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        camera_thread.start()
        
        # Start GUI update loop
        self.gui_update_loop()
        
    def camera_loop(self):
        """Camera processing loop"""
        try:
            self.initialize_camera()
            self.initialize_vitallens()
            
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                    
                heart_rate, confidence = self.process_frame(frame)
                
                # Update GUI in main thread
                if self.root:
                    self.root.after(0, self.update_heart_rate_display, heart_rate, confidence)
                    self.root.after(0, self.update_pattern_plot)
                    
        except Exception as e:
            print(f"Camera error: {e}")
            
    def gui_update_loop(self):
        """GUI update loop"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Restore original file open function
        builtins.open = builtins.__dict__.get('open', open)

def main():
    """Main function"""
    print("🫀 Heart Rate Lock System")
    print("=" * 40)
    
    lock_system = HeartRateLock()
    
    try:
        lock_system.run_gui()
    except Exception as e:
        print(f"❌ Error: {e}")
        lock_system.cleanup()

if __name__ == "__main__":
    main()