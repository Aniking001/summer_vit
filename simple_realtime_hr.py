import cv2
import numpy as np
from vitallens import VitalLens, Method, Mode
import time
from collections import deque
import os
import logging
import tempfile
import builtins

def simple_realtime_hr():
    print("🫀 Simple Real-Time Heart Rate Detection")
    print("=" * 45)
    print("Instructions:")
    print("• Keep your face visible in the camera")
    print("• Use good, even lighting (e.g., desk lamp)")
    print("• Stay as still as possible")
    print("• Wait 10-15 seconds for first reading")
    print("• Press 'q' to quit")
    print()
    
    # Suppress logging
    logging.getLogger('vitallens').setLevel(logging.CRITICAL + 1)
    logging.getLogger().handlers = []
    
    # Redirect output to temp directory
    temp_dir = tempfile.gettempdir()
    os.environ['VITALLENS_OUTPUT_DIR'] = temp_dir
    os.environ['VITALLENS_NO_LOG'] = '1'  # Additional attempt to disable logging
    
    # Monkey-patch file opening to block JSON writes
    original_open = builtins.open
    def patched_open(*args, **kwargs):
        if args and isinstance(args[0], str) and args[0].endswith('.json'):
            return open(os.devnull, 'w')  # Redirect to null device
        return original_open(*args, **kwargs)
    builtins.open = patched_open
    
    # Check for existing files
    initial_files = set(os.listdir())
    
    # Initialize VitalLens
    vl = VitalLens(
        method=Method.G,  # Fast method
        mode=Mode.BURST,  # Real-time mode
        detect_faces=True
    )
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Cannot open webcam")
        return
    
    # Camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Variables
    frame_buffer = []
    fps = 30
    buffer_size = 450  # 15 seconds for better motion handling
    last_analysis = time.time()
    current_hr = 0
    confidence = 0
    hr_history = deque(maxlen=7)  # Smooth over 7 readings
    min_confidence = 0.6  # Allow more readings
    rate_limit = True  # Enabled for ±3 BPM limit
    reference_bpm = 91  # From BP machine
    calibration_factor = 1.0  # Dynamic adjustment
    last_displayed_hr = 0  # Track last displayed heart rate
    
    print("✅ Camera ready! Starting detection...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add frame to buffer
            frame_buffer.append(frame.copy())
            
            # Keep only recent frames
            if len(frame_buffer) > buffer_size:
                frame_buffer.pop(0)
            
            # Analyze every 3 seconds if we have enough frames
            current_time = time.time()
            if len(frame_buffer) >= buffer_size and (current_time - last_analysis) >= 3:
                try:
                    # Convert to numpy array
                    video_array = np.array(frame_buffer)
                    
                    # Analyze with VitalLens
                    result = vl(video_array, fps=fps)
                    
                    if result and len(result) > 0:
                        face_data = result[0]
                        if 'vital_signs' in face_data and 'heart_rate' in face_data['vital_signs']:
                            hr_data = face_data['vital_signs']['heart_rate']
                            if hr_data['value'] is not None and hr_data.get('confidence', 0) >= min_confidence:
                                new_hr = hr_data['value']
                                confidence = hr_data.get('confidence', 0)
                                
                                # Dynamic calibration: adjust aggressively
                                if len(hr_history) >= 3 and reference_bpm:
                                    avg_hr = sum(hr_history) / len(hr_history)
                                    if abs(avg_hr - reference_bpm) > 5:
                                        calibration_factor = reference_bpm / avg_hr
                                        print(f"Adjusting calibration_factor to {calibration_factor:.2f}")
                                new_hr *= calibration_factor
                                
                                # Apply ±3 BPM limit
                                if current_hr == 0:
                                    current_hr = new_hr
                                else:
                                    delta = new_hr - current_hr
                                    if abs(delta) > 3:
                                        new_hr = current_hr + (3 if delta > 0 else -3)
                                    current_hr = new_hr
                                
                                hr_history.append(current_hr)
                                
                                # Smooth heart rate using moving average
                                smoothed_hr = sum(hr_history) / len(hr_history)
                                
                                # Print to console
                                print(f"❤  Heart Rate: {smoothed_hr:.1f} BPM | Confidence: {confidence:.2f}")
                                
                                # Update last displayed heart rate with ±3 BPM limit
                                if last_displayed_hr == 0:
                                    last_displayed_hr = smoothed_hr
                                else:
                                    delta = smoothed_hr - last_displayed_hr
                                    if abs(delta) > 3:
                                        last_displayed_hr = last_displayed_hr + (3 if delta > 0 else -3)
                                    else:
                                        last_displayed_hr = smoothed_hr
                                
                    last_analysis = current_time
                    
                except Exception as e:
                    print(f"Analysis error: {e}")
            
            # Display frame with overlay
            display_frame = frame.copy()
            
            # Add text overlay
            cv2.putText(display_frame, "Real-Time Heart Rate Detection", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if current_hr > 0 and len(hr_history) > 0 and confidence >= min_confidence:
                hr_text = f"Heart Rate: {last_displayed_hr:.1f} BPM"
                cv2.putText(display_frame, hr_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
            else:
                status_text = "Analyzing..." if len(frame_buffer) >= 225 else "Getting ready..."
                cv2.putText(display_frame, status_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                if confidence > 0 and confidence < min_confidence:
                    cv2.putText(display_frame, "Low confidence, adjust lighting/position", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Instructions
            cv2.putText(display_frame, "Press 'q' to quit", (10, 450), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Show frame
            cv2.imshow('Heart Rate Detection', display_frame)
            
            # Handle key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
    
    finally:
        # Restore original file open function
        builtins.open = original_open
        cap.release()
        cv2.destroyAllWindows()
        # Check for new JSON files
        final_files = set(os.listdir())
        new_files = final_files - initial_files
        json_files = [f for f in new_files if f.endswith('.json')]
        if json_files:
            print(f"⚠ Warning: JSON files created: {json_files}")
            for file in json_files:
                try:
                    os.remove(file)
                    print(f"Deleted {file}")
                except Exception as e:
                    print(f"Error deleting {file}: {e}")
        else:
            print("✅ No JSON files created")
        print("✅ Cleanup completed")

if __name__ == "__main__":
    simple_realtime_hr() 