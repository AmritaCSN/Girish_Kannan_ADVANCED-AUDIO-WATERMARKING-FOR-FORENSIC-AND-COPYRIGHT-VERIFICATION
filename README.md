# Girish_Kannan_ADVANCED-AUDIO-WATERMARKING-FOR-FORENSIC-AND-COPYRIGHT-VERIFICATION

This project implements a robust audio watermarking system using Spread Spectrum Modulation and Hamming Code for error correction. The system embeds a pseudo-random binary watermark directly into the time-frequency domain of an audio signal using the Short-Time Fourier Transform (STFT). It ensures robustness against audio distortions and extracts the embedded watermark with high accuracy.

Why This Project is Useful
Watermarking is essential for applications such as:
Copyright protection: Embed ownership information into audio files.
Forensic analysis: Prove audio authenticity and integrity.
Steganography: Securely convey information within audio signals.
But we are focusing on the application applicable which will help in forensic analysis.

This project demonstrates a systematic approach to embedding and extracting robust watermarks, featuring error-correcting capabilities using Hamming encoding.

Getting Started

To get started with the project, follow these steps:
1)Ensure the required Python libraries are installed. 
NumPy: Efficient numerical computation.
Librosa: Audio signal processing.
Matplotlib: Visualization of results.
SciPy: Signal filtering and transformations.
SoundFile: Audio file I/O.
Install the dependencies using pip:
pip install numpy librospip install numpy librosa matplotlib scipy soundfilea matplotlib scipy soundfile
2)Update the path of your audio file in the code:
PA_T_0000001.flac'(the audio file which you want to encrypt with watermark)

Quick Start (Default Settings)
1)Input Audio: Place the audio file in the specified directory.
2)Execution: Run the script with default settings. The system will generate a pseudo-random watermark.
3)Embed the watermark using default values for alpha and threshold.
4)Search for the best parameter combination to maximize accuracy.

Output:
A watermarked audio file saved as xyz.wav(any file name that you want).
The extracted watermark and accuracy displayed in the terminal.


Key Features
Error Correction: Uses Hamming encoding to enhance resilience against distortions.
Dynamic Parameter Search: Iteratively tests combinations of alpha (embedding strength) and threshold to optimize accuracy.
Denoising: Includes optional band-pass filtering for noise reduction.(optional)


Imput: Upload the audio in the code snippet
Output :
Watermark:
[ 1 -1 -1  1  1 -1 ...] ()
First 50 bits of the encoded watermark:
[1, 0, 1, 1, 1, 0, ...]
Best Alpha: 0.00500
Best Threshold: 0.02000
Best Accuracy: 97.50%


