import math
import streamlit as st
import librosa
import numpy as np
import parselmouth
import soundfile as sf
import io
import matplotlib.pyplot as plt

# Load audio file
file_path = '/Users/sanyakhurana/Documents/dheeme-dheeme-vocals-aalap.wav'
audio, sample_rate = librosa.load(file_path)

scales = {
    "C_major": [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88],
    "C_minor": [261.63, 293.66, 311.13, 349.23, 392.00, 415.30, 466.16],
    "C_harmonic_minor": [261.63, 293.66, 311.13, 349.23, 392.00, 415.30, 493.88],
    "C_melodic_minor": [261.63, 293.66, 311.13, 349.23, 392.00, 440.00, 493.88],

    "D_major": [293.66, 329.63, 369.99, 392.00, 440.00, 493.88, 554.37],
    "D_minor": [293.66, 329.63, 349.23, 392.00, 440.00, 466.16, 523.25],
    "D_harmonic_minor": [293.66, 329.63, 349.23, 392.00, 440.00, 466.16, 554.37],
    "D_melodic_minor": [293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 554.37],

    "E_major": [329.63, 369.99, 415.30, 440.00, 493.88, 554.37, 622.25],
    "E_minor": [329.63, 369.99, 392.00, 440.00, 493.88, 523.25, 587.33],
    "E_harmonic_minor": [329.63, 369.99, 392.00, 440.00, 493.88, 523.25, 622.25],
    "E_melodic_minor": [329.63, 369.99, 392.00, 440.00, 493.88, 554.37, 622.25],

    "F_major": [349.23, 392.00, 440.00, 466.16, 523.25, 587.33, 659.26],
    "F_minor": [349.23, 392.00, 415.30, 466.16, 523.25, 554.37, 622.25],
    "F_harmonic_minor": [349.23, 392.00, 415.30, 466.16, 523.25, 554.37, 659.26],
    "F_melodic_minor": [349.23, 392.00, 415.30, 466.16, 523.25, 587.33, 659.26],

    "G_major": [392.00, 440.00, 493.88, 523.25, 587.33, 659.26, 739.99],
    "G_minor": [392.00, 440.00, 466.16, 523.25, 587.33, 622.25, 698.46],
    "G_harmonic_minor": [392.00, 440.00, 466.16, 523.25, 587.33, 622.25, 739.99],
    "G_melodic_minor": [392.00, 440.00, 466.16, 523.25, 587.33, 659.26, 739.99],

    "A_major": [440.00, 493.88, 554.37, 587.33, 659.26, 739.99, 830.61],
    "A_minor": [440.00, 493.88, 523.25, 587.33, 659.26, 698.46, 783.99],
    "A_harmonic_minor": [440.00, 493.88, 523.25, 587.33, 659.26, 698.46, 830.61],
    "A_melodic_minor": [440.00, 493.88, 523.25, 587.33, 659.26, 739.99, 830.61],

    "B_major": [493.88, 554.37, 622.25, 659.26, 739.99, 830.61, 932.33],
    "B_minor": [493.88, 554.37, 587.33, 659.26, 739.99, 783.99, 880.00],
    "B_harmonic_minor": [493.88, 554.37, 587.33, 659.26, 739.99, 783.99, 932.33],
    "B_melodic_minor": [493.88, 554.37, 587.33, 659.26, 739.99, 830.61, 932.33],
}

# Select the desired scale
# Pitch detection function
# def detect_pitch(audio, sample_rate):
#     # Determine a suitable n_fft based on the audio length
#     n_fft = 2 ** math.floor(math.log2(len(audio)))
#     hop_length = n_fft // 4  # Typically, hop_length is set to n_fft / 4
#
#     print(f"Using n_fft = {n_fft}, hop_length = {hop_length}")
#
#     pitches, magnitudes = librosa.core.piptrack(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
#     pitch_values = []
#
#     for t in range(pitches.shape[1]):
#         index = magnitudes[:, t].argmax()
#         pitch = pitches[index, t]
#         if pitch > 0:
#             pitch_values.append(round(pitch, 2))
#
#     return pitch_values

def detect_pitch(audio, sample_rate):
    n_fft = 4096  # Reasonable FFT size for pitch detection
    #n_fft = 2 ** math.floor(math.log2(len(audio)))
    hop_length = n_fft // 8  # Finer time resolution

    print(f"Using n_fft = {n_fft}, hop_length = {hop_length}")

    pitches, magnitudes = librosa.piptrack(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
    pitch_values = []

    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_values.append(round(pitch, 2))

    return pitch_values


# Function to correct pitch to the nearest note in the selected scale
# def correct_pitch(pitch, scale):
#     if pitch == 0:
#         return 0
#     nearest_pitch = min(scale, key=lambda x: abs(x - pitch))
#     return nearest_pitch

def correct_pitch(pitch, scale, blend_ratio=0.5):
    if pitch == 0:
        return 0

    # Tolerance in Hz

    nearest_pitch = min(scale, key=lambda x: abs(x - pitch))

    # Blend detected and corrected pitch
    corrected_pitch = (1 - blend_ratio) * pitch + blend_ratio * nearest_pitch

    return corrected_pitch


# Function to visualize pitch contours
def visualize_pitch(audio, corrected_audio, sample_rate, original_pitches, corrected_pitches):
    # Create time axis for the pitch values
    duration = len(audio) / sample_rate
    time_axis = np.linspace(0, duration, len(original_pitches))

    print("Detected Pitches:", original_pitches[:20])


    # Plot the original and corrected pitch contours
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, original_pitches, label='Original Pitch', color='blue')
    plt.plot(time_axis, corrected_pitches, label='Corrected Pitch', color='red', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch (Hz)')
    plt.title('Pitch Contour Comparison')
    plt.legend()
    st.pyplot(plt)

# Function to apply pitch correction to segments of the audio
def apply_pitch_correction(audio, sample_rate, selected_scale, blend_ratio, blend_factor):

    detected_pitches = detect_pitch(audio, sample_rate)
    corrected_pitches = [correct_pitch(pitch, selected_scale, blend_ratio=blend_ratio) for pitch in detected_pitches]
    if not detected_pitches:
        print("No pitches detected. Skipping pitch correction.")
        return audio

    # Convert numpy array to Parselmouth Sound object
    sound = parselmouth.Sound(audio, sampling_frequency=sample_rate)

    # Create a Manipulation object with higher time resolution
    duration = sound.get_total_duration()
    manipulation = parselmouth.praat.call(sound, "To Manipulation", 0.002, 50, 600)

    # Extract the pitch tier
    pitch_tier = parselmouth.praat.call(manipulation, "Extract pitch tier")

    # Modify the pitch tier based on detected and corrected pitches with blending
    for i, (detected_pitch, corrected_pitch) in enumerate(zip(detected_pitches, corrected_pitches)):
        if detected_pitch > 0 and corrected_pitch > 0:
            time = i * (duration / len(detected_pitches))

            corrected_pitch = float(corrected_pitch)
            # Calculate the blended pitch and ensure it's a native Python float
            #blended_pitch = float((1 - blend_ratio) * detected_pitch + blend_ratio * corrected_pitch)
            # Remove existing pitch points around this time
            parselmouth.praat.call(pitch_tier, "Remove points between", time - 0.01, time + 0.01)
            # Add the blended pitch point
            parselmouth.praat.call(pitch_tier, "Add point", time, corrected_pitch)

    # Replace the pitch tier in the manipulation object
    parselmouth.praat.call([manipulation, pitch_tier], "Replace pitch tier")

    # Resynthesize the sound with the new pitch tier
    corrected_sound = parselmouth.praat.call(manipulation, "Get resynthesis (overlap-add)")

    # Convert the corrected sound back to a numpy array
    corrected_audio = corrected_sound.values[0]

    # Blend the original and corrected audio
    final_audio = (1 - blend_factor) * corrected_audio + blend_factor * audio

    # Normalize the final audio
    final_audio = final_audio / np.max(np.abs(final_audio))

    return final_audio

# Streamlit UI
st.title("AI-Powered Auto-Tuner")

# File upload
uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file is not None:
    # Load the audio file
    audio, sample_rate = librosa.load(uploaded_file, sr=None)
    st.audio(uploaded_file, format="audio/wav")

    # Select scale
    selected_scale_name = st.selectbox("Select Scale", list(scales.keys()))
    selected_scale = scales[selected_scale_name]

    # Adjust blend ratio (correction intensity)
    blend_ratio = st.slider("Correction Intensity (Blend Ratio)", 0.0, 1.0, 0.5)

    # Adjust blend factor (naturalness)
    blend_factor = st.slider("Naturalness (Blend Factor)", 0.0, 1.0, 0.7)

    # Apply pitch correction
    if st.button("Apply Pitch Correction"):
        with st.spinner("Processing..."):
            original_pitches = detect_pitch(audio, sample_rate)
            corrected_audio = apply_pitch_correction(audio, sample_rate, selected_scale, blend_ratio, blend_factor)

            corrected_pitches = detect_pitch(corrected_audio, sample_rate)
            # Save corrected audio to a buffer
            buffer = io.BytesIO()
            sf.write(buffer, corrected_audio, sample_rate, format="WAV")
            buffer.seek(0)

            st.success("Pitch correction applied successfully!")

            # Play the corrected audio in the browser
            st.audio(buffer, format="audio/wav")

            # Download button for corrected audio
            st.download_button(
                label="Download Corrected Audio",
                data=buffer,
                file_name="corrected_audio.wav",
                mime="audio/wav"
            )
            # Visualize pitch changes
            visualize_pitch(audio, corrected_audio, sample_rate, original_pitches, corrected_pitches)
