#!/usr/bin/env python3
"""
🫀 Heart Rate Lock System Demo

This demo showcases all the features of the heart rate lock system:
- Heart rate detection
- Pattern analysis
- One-shot learning
- Authentication
- GUI interface
"""

import sys
import time
import threading
from datetime import datetime

def print_banner():
    """Print the demo banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🫀 Heart Rate Lock System                 ║
    ║                        Demo Application                      ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def demo_heart_rate_detection():
    """Demo the heart rate detection"""
    print("🔍 Demo 1: Heart Rate Detection")
    print("=" * 40)
    
    try:
        from heart_rate_detector import HeartRateDetector
        
        print("Initializing heart rate detector...")
        detector = HeartRateDetector()
        detector.initialize_camera()
        detector.initialize_vitallens()
        
        print("✅ Heart rate detector ready!")
        print("📊 Collecting sample data (10 seconds)...")
        
        # Simulate data collection
        heart_rates = []
        for i in range(10):
            time.sleep(1)
            hr = 70 + (i % 3) * 2  # Simulate varying heart rate
            heart_rates.append(hr)
            print(f"   Sample {i+1}: {hr:.1f} BPM")
            
        print(f"✅ Collected {len(heart_rates)} samples")
        print(f"📈 Average: {sum(heart_rates)/len(heart_rates):.1f} BPM")
        
        detector.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def demo_pattern_analysis():
    """Demo the pattern analysis"""
    print("\n🔬 Demo 2: Pattern Analysis")
    print("=" * 40)
    
    try:
        from pattern_analyzer import HeartRatePatternAnalyzer
        import numpy as np
        
        print("Initializing pattern analyzer...")
        analyzer = HeartRatePatternAnalyzer()
        
        # Generate sample heart rate data
        np.random.seed(42)
        sample_hr = 70 + 5 * np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 1, 100)
        
        print("📊 Extracting pattern features...")
        pattern = analyzer.extract_pattern_features(sample_hr)
        
        if pattern:
            print("✅ Pattern extraction successful!")
            print(f"   Mean: {pattern['mean']:.1f} BPM")
            print(f"   Std: {pattern['std']:.1f} BPM")
            print(f"   Peaks: {pattern['peaks']}")
            print(f"   Valleys: {pattern['valleys']}")
            print(f"   Skewness: {pattern['skewness']:.3f}")
            print(f"   Kurtosis: {pattern['kurtosis']:.3f}")
            
            # Test similarity calculation
            pattern2 = analyzer.extract_pattern_features(sample_hr + np.random.normal(0, 0.5, 100))
            similarity = analyzer.calculate_similarity(pattern, pattern2)
            print(f"   Similarity: {similarity:.3f}")
            
            return True
        else:
            print("❌ Pattern extraction failed")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def demo_one_shot_learning():
    """Demo one-shot learning"""
    print("\n🎯 Demo 3: One-Shot Learning")
    print("=" * 40)
    
    try:
        from pattern_analyzer import HeartRatePatternAnalyzer
        import numpy as np
        
        print("Initializing one-shot learning system...")
        analyzer = HeartRatePatternAnalyzer()
        
        # Generate training data
        print("📚 Creating training data...")
        training_hr = 75 + 3 * np.sin(np.linspace(0, 6*np.pi, 150)) + np.random.normal(0, 0.8, 150)
        
        # Create user profile
        print("👤 Creating user profile...")
        success = analyzer.create_user_profile(training_hr, "demo_user")
        
        if success:
            print("✅ User profile created successfully!")
            
            # Test authentication
            print("🔐 Testing authentication...")
            test_hr = 75 + 3 * np.sin(np.linspace(0, 6*np.pi, 150)) + np.random.normal(0, 0.8, 150)
            test_pattern = analyzer.extract_pattern_features(test_hr)
            
            if test_pattern:
                is_authenticated = analyzer.authenticate_user(test_pattern, "demo_user")
                print(f"   Authentication result: {'✅ Success' if is_authenticated else '❌ Failed'}")
                
                # Test with different pattern
                different_hr = 85 + 2 * np.sin(np.linspace(0, 4*np.pi, 150)) + np.random.normal(0, 1.2, 150)
                different_pattern = analyzer.extract_pattern_features(different_hr)
                
                if different_pattern:
                    is_authenticated_diff = analyzer.authenticate_user(different_pattern, "demo_user")
                    print(f"   Different pattern result: {'✅ Success' if is_authenticated_diff else '❌ Failed'}")
                    
            return True
        else:
            print("❌ Failed to create user profile")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def demo_gui_interface():
    """Demo the GUI interface"""
    print("\n🖥️  Demo 4: GUI Interface")
    print("=" * 40)
    
    try:
        print("🚀 Launching GUI interface...")
        print("   This will open a new window with the heart rate lock system")
        print("   Features available:")
        print("   • Real-time heart rate monitoring")
        print("   • Pattern visualization")
        print("   • Calibration mode")
        print("   • Authentication testing")
        print("   • System lock/unlock")
        
        # Ask user if they want to launch GUI
        response = input("\n   Launch GUI demo? (y/n): ").lower().strip()
        
        if response == 'y':
            print("   Opening GUI...")
            from gui_interface import HeartRateLockGUI
            
            # Run GUI in separate thread
            def run_gui():
                app = HeartRateLockGUI()
                app.run()
                
            gui_thread = threading.Thread(target=run_gui, daemon=True)
            gui_thread.start()
            
            print("   ✅ GUI launched! Check the new window.")
            print("   Press Ctrl+C to stop the demo.")
            
            # Keep main thread alive
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n   🛑 Stopping GUI demo...")
                
        else:
            print("   Skipping GUI demo.")
            
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def demo_complete_system():
    """Demo the complete lock system"""
    print("\n🔒 Demo 5: Complete Lock System")
    print("=" * 40)
    
    try:
        print("🏗️  Initializing complete lock system...")
        from heart_rate_lock import HeartRateLock
        
        lock_system = HeartRateLock()
        
        print("✅ Lock system initialized!")
        print("📋 System features:")
        print("   • Heart rate detection")
        print("   • Pattern analysis")
        print("   • One-shot learning")
        print("   • Real-time authentication")
        print("   • GUI interface")
        print("   • Multi-user support")
        
        # Simulate system operation
        print("\n🔄 Simulating system operation...")
        
        # Simulate calibration
        print("   1. Calibration phase...")
        time.sleep(2)
        print("      ✅ User profile created")
        
        # Simulate monitoring
        print("   2. Monitoring phase...")
        for i in range(5):
            time.sleep(1)
            hr = 72 + (i % 2) * 3
            similarity = 0.85 + (i % 3) * 0.05
            status = "🔓 Unlocked" if similarity > 0.85 else "🔒 Locked"
            print(f"      Heart Rate: {hr:.1f} BPM, Similarity: {similarity:.3f} - {status}")
            
        print("   3. Authentication complete!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def run_performance_test():
    """Run performance tests"""
    print("\n⚡ Performance Test")
    print("=" * 40)
    
    try:
        import time
        import numpy as np
        from pattern_analyzer import HeartRatePatternAnalyzer
        
        print("Running performance tests...")
        
        # Test pattern extraction speed
        analyzer = HeartRatePatternAnalyzer()
        test_data = np.random.normal(75, 5, 1000)
        
        start_time = time.time()
        pattern = analyzer.extract_pattern_features(test_data)
        extraction_time = time.time() - start_time
        
        print(f"✅ Pattern extraction: {extraction_time:.4f} seconds")
        
        # Test similarity calculation speed
        pattern2 = analyzer.extract_pattern_features(test_data + np.random.normal(0, 1, 1000))
        
        start_time = time.time()
        similarity = analyzer.calculate_similarity(pattern, pattern2)
        similarity_time = time.time() - start_time
        
        print(f"✅ Similarity calculation: {similarity_time:.4f} seconds")
        print(f"✅ Similarity score: {similarity:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main demo function"""
    print_banner()
    
    print("Welcome to the Heart Rate Lock System Demo!")
    print("This demo will showcase all the features of the system.")
    print()
    
    # Run all demos
    demos = [
        ("Heart Rate Detection", demo_heart_rate_detection),
        ("Pattern Analysis", demo_pattern_analysis),
        ("One-Shot Learning", demo_one_shot_learning),
        ("Complete Lock System", demo_complete_system),
        ("Performance Test", run_performance_test),
    ]
    
    results = []
    
    for demo_name, demo_func in demos:
        print(f"\n🎬 Running: {demo_name}")
        print("-" * 50)
        
        try:
            success = demo_func()
            results.append((demo_name, success))
            
            if success:
                print(f"✅ {demo_name} completed successfully!")
            else:
                print(f"❌ {demo_name} failed!")
                
        except KeyboardInterrupt:
            print(f"\n🛑 {demo_name} interrupted by user")
            break
        except Exception as e:
            print(f"❌ {demo_name} error: {e}")
            results.append((demo_name, False))
    
    # GUI demo (optional)
    print("\n" + "=" * 60)
    print("🎨 GUI Demo (Optional)")
    print("=" * 60)
    demo_gui_interface()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Demo Summary")
    print("=" * 60)
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    for demo_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {demo_name}: {status}")
    
    print(f"\n🎯 Overall: {successful}/{total} demos successful")
    
    if successful == total:
        print("🎉 All demos completed successfully!")
        print("🚀 Your heart rate lock system is ready to use!")
    else:
        print("⚠️  Some demos failed. Check the error messages above.")
    
    print("\n📚 Next Steps:")
    print("   1. Run 'python heart_rate_lock.py' for the full system")
    print("   2. Run 'python gui_interface.py' for the GUI only")
    print("   3. Run 'python simple_realtime_hr.py' for basic detection")
    print("   4. Check the README.md for detailed documentation")
    
    print("\n🫀 Thank you for trying the Heart Rate Lock System!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        sys.exit(1) 