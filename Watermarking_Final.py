import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from scipy import signal

# Load the audio file
audio_file = 'E:/chaanged/PA_T_0000001.flac'
y, sr = librosa.load(audio_file, sr=None)

# Perform STFT to convert to time-frequency domain
D = librosa.stft(y)
magnitude, phase = np.abs(D), np.angle(D)

# Generate pseudo-random binary watermark
np.random.seed(42)  # For reproducibility
watermark_length = magnitude.shape[1]
watermark = np.random.choice([1, -1], watermark_length)

# Print the watermark
print("Watermark:")
print(watermark)

# Hamming Code for encoding the watermark
def hamming_encode(data_bits):
    if len(data_bits) % 4 != 0:
        padding_length = 4 - (len(data_bits) % 4)
        data_bits = np.append(data_bits, np.zeros(padding_length, dtype=int))
    
    encoded_bits = []
    for i in range(0, len(data_bits), 4):
        block = data_bits[i:i+4]
        P1 = (block[0] + block[1] + block[3]) % 2
        P2 = (block[0] + block[2] + block[3]) % 2
        P3 = (block[1] + block[2] + block[3]) % 2
        encoded_bits.extend([P1, P2, block[0], P3, block[1], block[2], block[3]])
    return encoded_bits

# Encode watermark using Hamming code
encoded_watermark = hamming_encode(watermark)

# Print the first 50 bits of the encoded watermark
print("First 50 bits of the encoded watermark:")
print(encoded_watermark[:50])

# Denoise the audio using a simple band-pass filter (if needed)
def denoise_audio(y, sr, lowcut=50.0, highcut=5000.0):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(1, [low, high], btype='band')
    y_denoised = signal.filtfilt(b, a, y)
    return y_denoised

# Denoise the audio before processing
y_denoised = denoise_audio(y, sr)

# Perform STFT on the denoised signal
D_denoised = librosa.stft(y_denoised)
magnitude_denoised, phase_denoised = np.abs(D_denoised), np.angle(D_denoised)

# Embed Watermark and Calculate Accuracy
def embed_and_extract(alpha, threshold):
    # Embed watermark using Spread Spectrum
    magnitude_copy = magnitude.copy()
    for i in range(magnitude.shape[1]):
        magnitude_copy[:, i] += alpha * encoded_watermark[i % len(encoded_watermark)] * np.max(magnitude_copy[:, i])

    # Reconstruct the watermarked signal
    watermarked_D = magnitude_copy * np.exp(1j * phase)
    watermarked_audio = librosa.istft(watermarked_D)

    # Save the watermarked audio
    sf.write('E:/chaanged/watermarked_audio_with_ecc.wav', watermarked_audio, sr)

    # Extract Watermark
    D_extracted = librosa.stft(watermarked_audio)
    magnitude_extracted = np.abs(D_extracted)

    # Compute the difference
    difference = magnitude_extracted - magnitude

    # Threshold-based detection
    detected_watermark = np.sign(np.sum(difference > threshold, axis=0))
    detected_watermark[detected_watermark == 0] = -1  # Map zeros to -1 for comparison

    # Decode the extracted watermark
    def hamming_decode(encoded_bits):
        decoded_bits = []
        for i in range(0, len(encoded_bits), 7):
            block = encoded_bits[i:i+7]
            if len(block) == 7:  # Ensure we have 7 bits to unpack
                P1, P2, D1, P3, D2, D3, D4 = block
                check_P1 = (D1 + D2 + D4) % 2
                check_P2 = (D1 + D3 + D4) % 2
                check_P3 = (D2 + D3 + D4) % 2
                error_pos = check_P1 * 1 + check_P2 * 2 + check_P3 * 4
                if error_pos:
                    block[error_pos - 1] = 1 - block[error_pos - 1]
                decoded_bits.append([D1, D2, D3, D4])
        return np.array(decoded_bits).flatten()

    decoded_watermark = hamming_decode(detected_watermark.flatten())

    # Calculate accuracy
    def calculate_accuracy(original, detected):
        length = min(len(original), len(detected))
        correct_bits = np.sum(original[:length] == detected[:length])
        accuracy = (correct_bits / length) * 100
        return accuracy

    accuracy = calculate_accuracy(encoded_watermark, decoded_watermark)
    return accuracy

# Loop over different values of alpha and threshold to find the best combination
best_accuracy = 0
best_alpha = 0
best_threshold = 0

# Range of alpha values from 0.1 to 0.001 (smaller steps for better refinement)
alphas = np.linspace(0.1, 0.001, num=100)  # 100 values between 0.1 and 0.001

# Range of threshold values from 0.001 to 0.05 (smaller steps for better refinement)
thresholds = np.linspace(0.001, 0.05, num=50)  # 50 values between 0.001 and 0.05

# Perform the search over both ranges
for alpha in alphas:
    for threshold in thresholds:
        accuracy = embed_and_extract(alpha, threshold)

        # Update best accuracy and parameters if the current one is better
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_alpha = alpha
            best_threshold = threshold

# Display the best combination
print(f"\nBest Alpha: {best_alpha:.5f}, Best Threshold: {best_threshold:.5f}, Best Accuracy: {best_accuracy:.2f}%")
