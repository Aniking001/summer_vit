import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
from matplotlib.figure import Figure
import numpy as np
import threading
import time
from datetime import datetime
import cv2
from PIL import Image, ImageTk
import os

class HeartRateLockGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🫀 Heart Rate Lock System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # System state
        self.is_locked = True
        self.is_calibrating = False
        self.is_monitoring = False
        self.current_heart_rate = None
        self.confidence = 0.0
        self.pattern_data = []
        self.similarity_score = 0.0
        
        # GUI components
        self.status_label = None
        self.heart_rate_label = None
        self.confidence_label = None
        self.similarity_label = None
        self.calibrate_btn = None
        self.lock_btn = None
        self.unlock_btn = None
        self.monitor_btn = None
        self.canvas = None
        self.ax = None
        self.ani = None
        
        # Camera components
        self.camera_label = None
        self.cap = None
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the main GUI"""
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🫀 Heart Rate Lock System", 
                               font=('Arial', 20, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Left panel - Controls
        self.setup_control_panel(main_frame)
        
        # Center panel - Status and Camera
        self.setup_status_panel(main_frame)
        
        # Right panel - Visualization
        self.setup_visualization_panel(main_frame)
        
        # Bottom panel - Log
        self.setup_log_panel(main_frame)
        
    def setup_control_panel(self, parent):
        """Setup the control panel"""
        control_frame = ttk.LabelFrame(parent, text="Controls", padding="15")
        control_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        
        # Status indicators
        status_frame = ttk.LabelFrame(control_frame, text="System Status", padding="10")
        status_frame.pack(fill="x", pady=(0, 15))
        
        self.status_label = ttk.Label(status_frame, text="🔒 System Locked", 
                                     font=('Arial', 12, 'bold'))
        self.status_label.pack(pady=5)
        
        self.heart_rate_label = ttk.Label(status_frame, text="Heart Rate: -- BPM", 
                                         font=('Arial', 10))
        self.heart_rate_label.pack(pady=2)
        
        self.confidence_label = ttk.Label(status_frame, text="Confidence: --", 
                                         font=('Arial', 10))
        self.confidence_label.pack(pady=2)
        
        self.similarity_label = ttk.Label(status_frame, text="Similarity: --", 
                                         font=('Arial', 10))
        self.similarity_label.pack(pady=2)
        
        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill="x", pady=10)
        
        self.calibrate_btn = ttk.Button(button_frame, text="🔧 Calibrate", 
                                       command=self.start_calibration, style='Accent.TButton')
        self.calibrate_btn.pack(fill="x", pady=5)
        
        self.monitor_btn = ttk.Button(button_frame, text="📊 Start Monitoring", 
                                     command=self.toggle_monitoring)
        self.monitor_btn.pack(fill="x", pady=5)
        
        self.lock_btn = ttk.Button(button_frame, text="🔒 Lock System", 
                                  command=self.lock_system)
        self.lock_btn.pack(fill="x", pady=5)
        
        self.unlock_btn = ttk.Button(button_frame, text="🔓 Unlock System", 
                                    command=self.unlock_system)
        self.unlock_btn.pack(fill="x", pady=5)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(control_frame, text="Settings", padding="10")
        settings_frame.pack(fill="x", pady=(15, 0))
        
        ttk.Label(settings_frame, text="Similarity Threshold:").pack(anchor="w")
        self.threshold_var = tk.DoubleVar(value=0.85)
        threshold_scale = ttk.Scale(settings_frame, from_=0.5, to=1.0, 
                                   variable=self.threshold_var, orient="horizontal")
        threshold_scale.pack(fill="x", pady=5)
        
        ttk.Label(settings_frame, text="Min Confidence:").pack(anchor="w")
        self.confidence_var = tk.DoubleVar(value=0.6)
        confidence_scale = ttk.Scale(settings_frame, from_=0.1, to=1.0, 
                                    variable=self.confidence_var, orient="horizontal")
        confidence_scale.pack(fill="x", pady=5)
        
    def setup_status_panel(self, parent):
        """Setup the status and camera panel"""
        status_frame = ttk.LabelFrame(parent, text="Camera & Status", padding="15")
        status_frame.grid(row=1, column=1, sticky="nsew", padx=10)
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(1, weight=1)
        
        # Camera display
        self.camera_label = ttk.Label(status_frame, text="Camera not initialized", 
                                     font=('Arial', 12))
        self.camera_label.grid(row=0, column=0, pady=10)
        
        # Status text
        self.status_text = tk.Text(status_frame, height=10, width=40, 
                                  font=('Consolas', 9))
        self.status_text.grid(row=1, column=0, sticky="nsew", pady=10)
        
        # Scrollbar for status text
        scrollbar = ttk.Scrollbar(status_frame, orient="vertical", 
                                 command=self.status_text.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.status_text.configure(yscrollcommand=scrollbar.set)
        
    def setup_visualization_panel(self, parent):
        """Setup the visualization panel"""
        viz_frame = ttk.LabelFrame(parent, text="Real-time Visualization", padding="15")
        viz_frame.grid(row=1, column=2, sticky="nsew", padx=(10, 0))
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(0, weight=1)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        # Initialize plot
        self.ax.set_title("Heart Rate Pattern")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Heart Rate (BPM)")
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()
        
    def setup_log_panel(self, parent):
        """Setup the log panel"""
        log_frame = ttk.LabelFrame(parent, text="System Log", padding="15")
        log_frame.grid(row=2, column=0, columnspan=3, sticky="nsew", pady=(20, 0))
        
        # Log text area
        self.log_text = tk.Text(log_frame, height=8, font=('Consolas', 9))
        self.log_text.pack(side="left", fill="both", expand=True)
        
        # Scrollbar for log
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", 
                                     command=self.log_text.yview)
        log_scrollbar.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        # Clear log button
        clear_btn = ttk.Button(log_frame, text="Clear Log", 
                              command=self.clear_log)
        clear_btn.pack(side="bottom", pady=(5, 0))
        
    def log_message(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        
    def clear_log(self):
        """Clear the log"""
        self.log_text.delete(1.0, tk.END)
        
    def update_status(self, status, color="black"):
        """Update status display"""
        if self.status_label:
            self.status_label.config(text=status, foreground=color)
            
    def update_heart_rate_display(self, heart_rate, confidence):
        """Update heart rate display"""
        if heart_rate:
            self.current_heart_rate = heart_rate
            self.confidence = confidence
            
            if self.heart_rate_label:
                self.heart_rate_label.config(text=f"Heart Rate: {heart_rate:.1f} BPM")
                
            if self.confidence_label:
                self.confidence_label.config(text=f"Confidence: {confidence:.2f}")
                
    def update_similarity_display(self, similarity):
        """Update similarity display"""
        self.similarity_score = similarity
        if self.similarity_label:
            self.similarity_label.config(text=f"Similarity: {similarity:.3f}")
            
    def update_pattern_plot(self, pattern_data):
        """Update the pattern visualization"""
        if len(pattern_data) > 10:
            self.ax.clear()
            
            # Plot current pattern
            self.ax.plot(pattern_data, 'b-', label='Current Pattern', linewidth=2)
            
            # Add threshold line
            threshold = self.threshold_var.get()
            self.ax.axhline(y=threshold, color='r', linestyle='--', 
                           label=f'Threshold ({threshold})', alpha=0.7)
            
            self.ax.set_title("Heart Rate Pattern")
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("Heart Rate (BPM)")
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)
            
            self.canvas.draw()
            
    def start_calibration(self):
        """Start calibration process"""
        if messagebox.askyesno("Calibration", 
                              "This will create a new user profile.\n"
                              "Please stay still and look at the camera for 30 seconds.\n\n"
                              "Continue?"):
            self.is_calibrating = True
            self.update_status("🔄 Calibrating...", "orange")
            self.log_message("Starting calibration process...")
            
            # Disable buttons during calibration
            self.calibrate_btn.config(state='disabled')
            self.monitor_btn.config(state='disabled')
            
            # Start calibration in separate thread
            calibration_thread = threading.Thread(target=self.run_calibration, daemon=True)
            calibration_thread.start()
            
    def run_calibration(self):
        """Run calibration process"""
        try:
            # Initialize camera
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.log_message("❌ Error: Cannot open webcam")
                return
                
            # Camera settings
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Collect heart rate data
            heart_rates = []
            start_time = time.time()
            
            while time.time() - start_time < 30:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                    
                # Process frame (simplified for demo)
                # In real implementation, use VitalLens here
                heart_rate = self.simulate_heart_rate_detection(frame)
                
                if heart_rate:
                    heart_rates.append(heart_rate)
                    self.pattern_data.append(heart_rate)
                    
                # Update GUI
                self.root.after(0, self.update_heart_rate_display, heart_rate, 0.8)
                self.root.after(0, self.update_pattern_plot, self.pattern_data)
                self.root.after(0, self.log_message, 
                              f"Collected {len(heart_rates)} samples...")
                
                # Display camera frame
                if frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    frame_tk = ImageTk.PhotoImage(frame_pil)
                    
                    self.root.after(0, lambda: self.camera_label.config(image=frame_tk))
                    self.root.after(0, lambda: setattr(self, 'current_frame', frame_tk))
                    
            # Create user profile
            if len(heart_rates) >= 20:
                self.root.after(0, self.log_message, 
                              f"✅ Calibration completed with {len(heart_rates)} samples")
                self.root.after(0, self.update_status, "✅ Calibration Complete", "green")
                self.is_locked = False
            else:
                self.root.after(0, self.log_message, 
                              "❌ Calibration failed - insufficient data")
                self.root.after(0, self.update_status, "❌ Calibration Failed", "red")
                
        except Exception as e:
            self.root.after(0, self.log_message, f"❌ Calibration error: {e}")
            self.root.after(0, self.update_status, "❌ Calibration Error", "red")
            
        finally:
            self.is_calibrating = False
            self.root.after(0, self.update_button_states)
            if self.cap:
                self.cap.release()
                
    def simulate_heart_rate_detection(self, frame):
        """Simulate heart rate detection for demo purposes"""
        # In real implementation, use VitalLens here
        import random
        return 70 + random.uniform(-5, 5)
        
    def toggle_monitoring(self):
        """Toggle monitoring mode"""
        if not self.is_monitoring:
            self.start_monitoring()
        else:
            self.stop_monitoring()
            
    def start_monitoring(self):
        """Start monitoring mode"""
        self.is_monitoring = True
        self.monitor_btn.config(text="⏹ Stop Monitoring")
        self.update_status("📊 Monitoring Active", "blue")
        self.log_message("Started heart rate monitoring...")
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
        monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring mode"""
        self.is_monitoring = False
        self.monitor_btn.config(text="📊 Start Monitoring")
        self.update_status("⏹ Monitoring Stopped", "gray")
        self.log_message("Stopped heart rate monitoring...")
        
    def monitoring_loop(self):
        """Monitoring loop"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.log_message("❌ Error: Cannot open webcam")
                return
                
            while self.is_monitoring:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                    
                # Process frame
                heart_rate = self.simulate_heart_rate_detection(frame)
                
                if heart_rate:
                    self.pattern_data.append(heart_rate)
                    if len(self.pattern_data) > 100:
                        self.pattern_data.pop(0)
                        
                    # Calculate similarity (simplified)
                    similarity = self.calculate_similarity()
                    
                    # Update GUI
                    self.root.after(0, self.update_heart_rate_display, heart_rate, 0.8)
                    self.root.after(0, self.update_similarity_display, similarity)
                    self.root.after(0, self.update_pattern_plot, self.pattern_data)
                    
                    # Check authentication
                    if similarity >= self.threshold_var.get():
                        self.root.after(0, self.update_status, "🔓 Authenticated", "green")
                        self.is_locked = False
                    else:
                        self.root.after(0, self.update_status, "🔒 Not Authenticated", "red")
                        self.is_locked = True
                        
                time.sleep(0.1)  # 10 FPS
                
        except Exception as e:
            self.root.after(0, self.log_message, f"❌ Monitoring error: {e}")
        finally:
            if self.cap:
                self.cap.release()
                
    def calculate_similarity(self):
        """Calculate similarity score (simplified)"""
        if len(self.pattern_data) < 10:
            return 0.0
            
        # Simple similarity calculation for demo
        # In real implementation, use pattern analyzer
        mean_hr = np.mean(self.pattern_data)
        if 65 <= mean_hr <= 85:  # Normal heart rate range
            return 0.9
        else:
            return 0.3
            
    def lock_system(self):
        """Lock the system"""
        self.is_locked = True
        self.update_status("🔒 System Locked", "red")
        self.log_message("System locked")
        self.update_button_states()
        
    def unlock_system(self):
        """Unlock the system"""
        if not self.is_calibrating:
            self.is_locked = False
            self.update_status("🔓 System Unlocked", "green")
            self.log_message("System unlocked")
            self.update_button_states()
        else:
            messagebox.showwarning("Warning", "Please wait for calibration to complete!")
            
    def update_button_states(self):
        """Update button states based on system state"""
        if self.is_calibrating:
            self.calibrate_btn.config(state='disabled')
            self.monitor_btn.config(state='disabled')
        else:
            self.calibrate_btn.config(state='normal')
            self.monitor_btn.config(state='normal')
            
        if self.is_locked:
            self.lock_btn.config(state='disabled')
            self.unlock_btn.config(state='normal')
        else:
            self.lock_btn.config(state='normal')
            self.unlock_btn.config(state='disabled')
            
    def run(self):
        """Run the GUI"""
        try:
            self.log_message("🫀 Heart Rate Lock System started")
            self.log_message("Please calibrate to create your heart rate profile")
            self.update_button_states()
            self.root.mainloop()
        except KeyboardInterrupt:
            self.log_message("🛑 System shutdown requested")
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()

def main():
    """Main function"""
    app = HeartRateLockGUI()
    app.run()

if __name__ == "__main__":
    main() 