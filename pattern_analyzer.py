import numpy as np
from scipy.signal import savgol_filter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt

class HeartRatePatternAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.user_profiles = {}
        self.similarity_threshold = 0.85
        self.min_pattern_length = 20
        
    def extract_pattern_features(self, heart_rates):
        """Extract comprehensive features from heart rate pattern - FOCUSING ON RELATIVE PATTERNS, NOT ABSOLUTE VALUES"""
        if len(heart_rates) < self.min_pattern_length:
            return None
            
        # Convert to numpy array
        hr_array = np.array(heart_rates)
        
        # Apply smoothing
        if len(hr_array) > 5:
            hr_smoothed = savgol_filter(hr_array, min(5, len(hr_array)//2), 2)
        else:
            hr_smoothed = hr_array
            
        # NORMALIZE THE SIGNAL - This is key for pattern-based authentication
        hr_normalized = (hr_smoothed - np.mean(hr_smoothed)) / np.std(hr_smoothed)
        
        # Extract RELATIVE features (not absolute heart rate values)
        features = {
            # RELATIVE PATTERN FEATURES (not absolute values)
            'normalized_pattern': hr_normalized.tolist()[:100],  # Normalized pattern
            'relative_peaks': self._count_relative_peaks(hr_normalized),
            'relative_valleys': self._count_relative_valleys(hr_normalized),
            'zero_crossings': self._count_zero_crossings(hr_normalized),
            
            # PATTERN SHAPE FEATURES (independent of absolute values)
            'pattern_skewness': self._calculate_skewness(hr_normalized),
            'pattern_kurtosis': self._calculate_kurtosis(hr_normalized),
            'pattern_entropy': self._calculate_entropy(hr_normalized),
            'pattern_complexity': self._calculate_complexity(hr_normalized),
            
            # RHYTHM FEATURES (timing patterns, not values)
            'rhythm_regularity': self._calculate_rhythm_regularity(hr_normalized),
            'peak_spacing': self._calculate_peak_spacing(hr_normalized),
            'valley_spacing': self._calculate_valley_spacing(hr_normalized),
            
            # FREQUENCY DOMAIN FEATURES (relative frequencies)
            'fft_features': self._extract_relative_fft_features(hr_normalized),
            'autocorr_features': self._extract_autocorr_features(hr_normalized),
            
            # VARIABILITY FEATURES (pattern consistency)
            'variability_score': self._calculate_variability_score(hr_normalized),
            'stability_index': self._calculate_stability_index(hr_normalized),
            
            # WAVEFORM FEATURES (shape characteristics)
            'waveform_symmetry': self._calculate_waveform_symmetry(hr_normalized),
            'waveform_smoothness': self._calculate_waveform_smoothness(hr_normalized),
            
            # Store original stats for reference (but not used for authentication)
            'original_mean': np.mean(hr_smoothed),
            'original_std': np.std(hr_smoothed),
        }
        
        return features
        
    def _count_relative_peaks(self, signal):
        """Count peaks in the normalized signal"""
        if len(signal) < 3:
            return 0
            
        peaks = 0
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                peaks += 1
        return peaks
        
    def _count_relative_valleys(self, signal):
        """Count valleys in the normalized signal"""
        if len(signal) < 3:
            return 0
            
        valleys = 0
        for i in range(1, len(signal) - 1):
            if signal[i] < signal[i-1] and signal[i] < signal[i+1]:
                valleys += 1
        return valleys
        
    def _count_zero_crossings(self, signal):
        """Count zero crossings in the normalized signal"""
        if len(signal) < 2:
            return 0
            
        crossings = 0
        for i in range(1, len(signal)):
            if (signal[i] >= 0 and signal[i-1] < 0) or (signal[i] < 0 and signal[i-1] >= 0):
                crossings += 1
        return crossings
        
    def _calculate_skewness(self, signal):
        """Calculate skewness of the normalized signal"""
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            return 0
        return np.mean(((signal - mean) / std) ** 3)
        
    def _calculate_kurtosis(self, signal):
        """Calculate kurtosis of the normalized signal"""
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            return 0
        return np.mean(((signal - mean) / std) ** 4) - 3
        
    def _calculate_entropy(self, signal):
        """Calculate entropy of the signal pattern"""
        try:
            # Discretize signal into bins
            bins = np.histogram(signal, bins=20)[0]
            bins = bins[bins > 0]  # Remove zero bins
            if len(bins) == 0:
                return 0
            prob = bins / np.sum(bins)
            return -np.sum(prob * np.log(prob))
        except:
            return 0
            
    def _calculate_complexity(self, signal):
        """Calculate complexity of the signal pattern"""
        # Use sample entropy as complexity measure
        try:
            m = 2  # embedding dimension
            r = 0.2 * np.std(signal)  # tolerance
            
            complexity = 0
            for i in range(len(signal) - m):
                for j in range(i + 1, len(signal) - m):
                    if np.max(np.abs(signal[i:i+m] - signal[j:j+m])) <= r:
                        complexity += 1
            return complexity
        except:
            return 0
            
    def _calculate_rhythm_regularity(self, signal):
        """Calculate rhythm regularity of the pattern"""
        try:
            # Find peaks
            peaks = []
            for i in range(1, len(signal) - 1):
                if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                    peaks.append(i)
                    
            if len(peaks) < 2:
                return 0
                
            # Calculate intervals between peaks
            intervals = np.diff(peaks)
            return np.std(intervals) / np.mean(intervals)  # Coefficient of variation
        except:
            return 0
            
    def _calculate_peak_spacing(self, signal):
        """Calculate average spacing between peaks"""
        try:
            peaks = []
            for i in range(1, len(signal) - 1):
                if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                    peaks.append(i)
                    
            if len(peaks) < 2:
                return 0
                
            return np.mean(np.diff(peaks))
        except:
            return 0
            
    def _calculate_valley_spacing(self, signal):
        """Calculate average spacing between valleys"""
        try:
            valleys = []
            for i in range(1, len(signal) - 1):
                if signal[i] < signal[i-1] and signal[i] < signal[i+1]:
                    valleys.append(i)
                    
            if len(valleys) < 2:
                return 0
                
            return np.mean(np.diff(valleys))
        except:
            return 0
            
    def _extract_relative_fft_features(self, signal):
        """Extract FFT-based features from normalized signal"""
        try:
            fft = np.fft.fft(signal)
            magnitude = np.abs(fft)
            
            # Get dominant frequencies (relative to signal length)
            freqs = np.fft.fftfreq(len(signal))
            dominant_freq_idx = np.argmax(magnitude[1:len(magnitude)//2]) + 1
            dominant_freq = freqs[dominant_freq_idx]
            
            return {
                'dominant_freq_relative': dominant_freq,
                'fft_energy_relative': np.sum(magnitude**2),
                'fft_entropy_relative': -np.sum(magnitude * np.log(magnitude + 1e-10)),
                'spectral_centroid': np.sum(freqs * magnitude) / np.sum(magnitude),
                'spectral_bandwidth': np.sqrt(np.sum(((freqs - np.sum(freqs * magnitude) / np.sum(magnitude))**2) * magnitude) / np.sum(magnitude))
            }
        except:
            return {'dominant_freq_relative': 0, 'fft_energy_relative': 0, 'fft_entropy_relative': 0, 'spectral_centroid': 0, 'spectral_bandwidth': 0}
            
    def _extract_autocorr_features(self, signal):
        """Extract autocorrelation features from normalized signal"""
        try:
            autocorr = np.correlate(signal, signal, mode='full')
            autocorr = autocorr[len(signal)-1:]
            
            # Find first peak after lag 0
            peaks = []
            for i in range(1, len(autocorr)-1):
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                    peaks.append(i)
                    
            if peaks:
                first_peak_lag = peaks[0]
                autocorr_energy = np.sum(autocorr**2)
            else:
                first_peak_lag = 0
                autocorr_energy = 0
                
            return {
                'first_peak_lag_relative': first_peak_lag,
                'autocorr_energy_relative': autocorr_energy,
                'autocorr_decay_rate': self._calculate_autocorr_decay(autocorr)
            }
        except:
            return {'first_peak_lag_relative': 0, 'autocorr_energy_relative': 0, 'autocorr_decay_rate': 0}
            
    def _calculate_autocorr_decay(self, autocorr):
        """Calculate decay rate of autocorrelation"""
        try:
            # Find where autocorrelation drops to 50% of max
            max_val = np.max(autocorr)
            half_max = max_val * 0.5
            
            for i in range(1, len(autocorr)):
                if autocorr[i] < half_max:
                    return i
            return len(autocorr)
        except:
            return 0
            
    def _calculate_variability_score(self, signal):
        """Calculate variability score of the pattern"""
        try:
            # Use coefficient of variation of first differences
            diff = np.diff(signal)
            return np.std(diff) / (np.mean(np.abs(diff)) + 1e-10)
        except:
            return 0
            
    def _calculate_stability_index(self, signal):
        """Calculate stability index of the pattern"""
        try:
            # Use inverse of variance of local means
            window_size = min(10, len(signal) // 4)
            local_means = []
            for i in range(0, len(signal) - window_size, window_size):
                local_means.append(np.mean(signal[i:i+window_size]))
            return 1 / (np.var(local_means) + 1e-10)
        except:
            return 0
            
    def _calculate_waveform_symmetry(self, signal):
        """Calculate symmetry of the waveform"""
        try:
            # Compare positive and negative parts
            positive_part = signal[signal > 0]
            negative_part = signal[signal < 0]
            
            if len(positive_part) == 0 or len(negative_part) == 0:
                return 0
                
            pos_std = np.std(positive_part)
            neg_std = np.std(negative_part)
            
            return 1 - abs(pos_std - neg_std) / (pos_std + neg_std + 1e-10)
        except:
            return 0
            
    def _calculate_waveform_smoothness(self, signal):
        """Calculate smoothness of the waveform"""
        try:
            # Use inverse of sum of squared second differences
            second_diff = np.diff(signal, n=2)
            return 1 / (np.sum(second_diff**2) + 1e-10)
        except:
            return 0
        
    def calculate_similarity(self, pattern1, pattern2):
        """Calculate similarity between two heart rate patterns - FOCUSING ON RELATIVE FEATURES"""
        if not pattern1 or not pattern2:
            return 0.0
            
        # Compare NORMALIZED pattern arrays (not absolute values)
        if len(pattern1['normalized_pattern']) > 0 and len(pattern2['normalized_pattern']) > 0:
            # Pad shorter pattern
            max_len = max(len(pattern1['normalized_pattern']), len(pattern2['normalized_pattern']))
            p1 = pattern1['normalized_pattern'] + [pattern1['normalized_pattern'][-1]] * (max_len - len(pattern1['normalized_pattern']))
            p2 = pattern2['normalized_pattern'] + [pattern2['normalized_pattern'][-1]] * (max_len - len(pattern2['normalized_pattern']))
            
            # Calculate cosine similarity of normalized patterns
            pattern_similarity = cosine_similarity([p1], [p2])[0][0]
            
            # Compare RELATIVE pattern features (not absolute heart rate values)
            relative_features1 = [
                pattern1['pattern_skewness'], pattern1['pattern_kurtosis'], 
                pattern1['pattern_entropy'], pattern1['pattern_complexity'],
                pattern1['rhythm_regularity'], pattern1['peak_spacing'], 
                pattern1['valley_spacing'], pattern1['variability_score'],
                pattern1['stability_index'], pattern1['waveform_symmetry'],
                pattern1['waveform_smoothness']
            ]
            relative_features2 = [
                pattern2['pattern_skewness'], pattern2['pattern_kurtosis'], 
                pattern2['pattern_entropy'], pattern2['pattern_complexity'],
                pattern2['rhythm_regularity'], pattern2['peak_spacing'], 
                pattern2['valley_spacing'], pattern2['variability_score'],
                pattern2['stability_index'], pattern2['waveform_symmetry'],
                pattern2['waveform_smoothness']
            ]
            
            feature_similarity = cosine_similarity([relative_features1], [relative_features2])[0][0]
            
            # Compare FFT features (relative frequencies)
            fft_features1 = [
                pattern1['fft_features']['dominant_freq_relative'],
                pattern1['fft_features']['fft_energy_relative'],
                pattern1['fft_features']['spectral_centroid'],
                pattern1['fft_features']['spectral_bandwidth']
            ]
            fft_features2 = [
                pattern2['fft_features']['dominant_freq_relative'],
                pattern2['fft_features']['fft_energy_relative'],
                pattern2['fft_features']['spectral_centroid'],
                pattern2['fft_features']['spectral_bandwidth']
            ]
            
            fft_similarity = cosine_similarity([fft_features1], [fft_features2])[0][0]
            
            # Weighted combination - FOCUSING ON RELATIVE PATTERNS
            final_similarity = (0.5 * pattern_similarity + 
                              0.3 * feature_similarity + 
                              0.2 * fft_similarity)
            
            return max(0, final_similarity)  # Ensure non-negative
            
        return 0.0
        
    def create_user_profile(self, heart_rates, user_id="default"):
        """Create user profile from heart rate data - BASED ON RELATIVE PATTERNS"""
        pattern = self.extract_pattern_features(heart_rates)
        if pattern:
            self.user_profiles[user_id] = {
                'pattern': pattern,
                'created_at': datetime.now().isoformat(),
                'heart_rate_count': len(heart_rates),
                'note': 'Profile based on RELATIVE heart rate patterns, not absolute values'
            }
            self.save_profiles()
            return True
        return False
        
    def authenticate_user(self, current_pattern, user_id="default"):
        """Authenticate user using RELATIVE heart rate patterns"""
        if user_id not in self.user_profiles:
            return False
            
        user_profile = self.user_profiles[user_id]['pattern']
        similarity = self.calculate_similarity(user_profile, current_pattern)
        return similarity >= self.similarity_threshold
        
    def save_profiles(self):
        """Save user profiles to file"""
        try:
            with open('heart_rate_profiles.pkl', 'wb') as f:
                pickle.dump(self.user_profiles, f)
            print("✅ User profiles saved (based on relative patterns)")
        except Exception as e:
            print(f"❌ Error saving profiles: {e}")
            
    def load_profiles(self):
        """Load user profiles from file"""
        try:
            if os.path.exists('heart_rate_profiles.pkl'):
                with open('heart_rate_profiles.pkl', 'rb') as f:
                    self.user_profiles = pickle.load(f)
                print(f"✅ Loaded {len(self.user_profiles)} user profiles (relative patterns)")
                return True
        except Exception as e:
            print(f"❌ Error loading profiles: {e}")
        return False
        
    def get_profile_info(self, user_id="default"):
        """Get information about a user profile"""
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            return {
                'created_at': profile['created_at'],
                'heart_rate_count': profile['heart_rate_count'],
                'note': profile.get('note', 'Relative pattern-based profile'),
                'pattern_stats': {
                    'skewness': profile['pattern']['pattern_skewness'],
                    'kurtosis': profile['pattern']['pattern_kurtosis'],
                    'entropy': profile['pattern']['pattern_entropy'],
                    'complexity': profile['pattern']['pattern_complexity']
                }
            }
        return None
        
    def visualize_pattern(self, heart_rates, title="Heart Rate Pattern (Relative)"):
        """Visualize heart rate pattern - SHOWING RELATIVE PATTERNS"""
        if len(heart_rates) < 10:
            print("❌ Insufficient data for visualization")
            return
            
        pattern = self.extract_pattern_features(heart_rates)
        if not pattern:
            return
            
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot raw heart rate data
        ax1.plot(heart_rates, 'b-', label='Raw Heart Rate', linewidth=1)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Heart Rate (BPM)')
        ax1.set_title(f'{title} - Raw Data')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot normalized pattern (what we actually use for authentication)
        normalized_pattern = pattern['normalized_pattern']
        ax2.plot(normalized_pattern, 'r-', label='Normalized Pattern (Used for Auth)', linewidth=2)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Normalized Value')
        ax2.set_title('Normalized Pattern (Relative Features)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot relative features
        features = ['Skewness', 'Kurtosis', 'Entropy', 'Complexity', 'Variability']
        values = [pattern['pattern_skewness'], pattern['pattern_kurtosis'], 
                 pattern['pattern_entropy'], pattern['pattern_complexity'], 
                 pattern['variability_score']]
        ax3.bar(features, values, color='skyblue')
        ax3.set_ylabel('Value')
        ax3.set_title('Relative Pattern Features (Not Absolute HR Values)')
        ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
    def compare_patterns(self, pattern1, pattern2, title1="Pattern 1", title2="Pattern 2"):
        """Compare two heart rate patterns - FOCUSING ON RELATIVE FEATURES"""
        if not pattern1 or not pattern2:
            print("❌ Invalid patterns for comparison")
            return
            
        similarity = self.calculate_similarity(pattern1, pattern2)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot normalized patterns (what we actually compare)
        ax1.plot(pattern1['normalized_pattern'], 'b-', label=title1, linewidth=2)
        ax1.plot(pattern2['normalized_pattern'], 'r-', label=title2, linewidth=2)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Normalized Value')
        ax1.set_title('Normalized Patterns (Relative Features)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot relative feature comparison
        features = ['Skewness', 'Kurtosis', 'Entropy', 'Complexity', 'Variability']
        p1_features = [pattern1['pattern_skewness'], pattern1['pattern_kurtosis'], 
                      pattern1['pattern_entropy'], pattern1['pattern_complexity'], 
                      pattern1['variability_score']]
        p2_features = [pattern2['pattern_skewness'], pattern2['pattern_kurtosis'], 
                      pattern2['pattern_entropy'], pattern2['pattern_complexity'], 
                      pattern2['variability_score']]
        
        x = np.arange(len(features))
        width = 0.35
        ax2.bar(x - width/2, p1_features, width, label=title1, color='blue', alpha=0.7)
        ax2.bar(x + width/2, p2_features, width, label=title2, color='red', alpha=0.7)
        ax2.set_xlabel('Features')
        ax2.set_ylabel('Value')
        ax2.set_title('Relative Feature Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(features)
        ax2.legend()
        
        plt.suptitle(f'Pattern Comparison - Similarity: {similarity:.3f} (Based on Relative Features)')
        plt.tight_layout()
        plt.show()
        
        return similarity

def test_pattern_analyzer():
    """Test the pattern analyzer - DEMONSTRATING RELATIVE PATTERN ANALYSIS"""
    analyzer = HeartRatePatternAnalyzer()
    
    # Generate sample heart rate data with different absolute values but similar patterns
    np.random.seed(42)
    base_pattern = np.sin(np.linspace(0, 4*np.pi, 100))
    
    # Pattern 1: Low heart rate (60-70 BPM)
    sample_hr1 = 65 + 3 * base_pattern + np.random.normal(0, 1, 100)
    
    # Pattern 2: High heart rate (90-100 BPM) but SAME RELATIVE PATTERN
    sample_hr2 = 95 + 3 * base_pattern + np.random.normal(0, 1, 100)
    
    # Pattern 3: Different pattern (different shape)
    sample_hr3 = 75 + 2 * np.sin(np.linspace(0, 2*np.pi, 100)) + np.random.normal(0, 1, 100)
    
    print("🫀 Heart Rate Pattern Analyzer Test (RELATIVE PATTERNS)")
    print("=" * 60)
    print("Testing authentication based on RELATIVE patterns, not absolute HR values")
    print()
    
    # Extract patterns
    pattern1 = analyzer.extract_pattern_features(sample_hr1)
    pattern2 = analyzer.extract_pattern_features(sample_hr2)
    pattern3 = analyzer.extract_pattern_features(sample_hr3)
    
    if pattern1 and pattern2 and pattern3:
        print("✅ Pattern extraction successful!")
        print(f"Pattern 1 (Low HR): Mean={pattern1['original_mean']:.1f} BPM")
        print(f"Pattern 2 (High HR): Mean={pattern2['original_mean']:.1f} BPM")
        print(f"Pattern 3 (Different): Mean={pattern3['original_mean']:.1f} BPM")
        print()
        
        # Test similarity between same pattern with different absolute values
        similarity_same_pattern = analyzer.calculate_similarity(pattern1, pattern2)
        print(f"Similarity (Same Pattern, Different HR): {similarity_same_pattern:.3f}")
        print("   → Should be HIGH (same relative pattern)")
        
        # Test similarity between different patterns
        similarity_diff_pattern = analyzer.calculate_similarity(pattern1, pattern3)
        print(f"Similarity (Different Patterns): {similarity_diff_pattern:.3f}")
        print("   → Should be LOW (different relative patterns)")
        
        # Create user profile
        if analyzer.create_user_profile(sample_hr1, "user1"):
            print("\n✅ User profile created (based on relative pattern)")
            
            # Test authentication with same pattern, different HR
            auth_same_pattern = analyzer.authenticate_user(pattern2, "user1")
            print(f"Auth (Same Pattern, Different HR): {'✅ Success' if auth_same_pattern else '❌ Failed'}")
            
            # Test authentication with different pattern
            auth_diff_pattern = analyzer.authenticate_user(pattern3, "user1")
            print(f"Auth (Different Pattern): {'✅ Success' if auth_diff_pattern else '❌ Failed'}")
            
        # Visualize patterns
        print("\n📊 Visualizing patterns...")
        analyzer.visualize_pattern(sample_hr1, "Low Heart Rate Pattern")
        analyzer.compare_patterns(pattern1, pattern2, "Low HR Pattern", "High HR Pattern (Same Relative Pattern)")
        
    else:
        print("❌ Pattern extraction failed")

if __name__ == "__main__":
    test_pattern_analyzer() 