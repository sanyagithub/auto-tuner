import streamlit as st
import librosa
import librosa.display
import numpy as np
import pyrubberband as pyrb
import noisereduce as nr
import soundfile as sf
import matplotlib.pyplot as plt
import io

# C Minor scale frequencies
C_MINOR_SCALE = [261.63, 293.66, 311.13, 349.23, 392.00, 415.30, 466.16]

# Function to visualize audio waveform
def visualize_audio(audio, sample_rate, title):
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(audio, sr=sample_rate)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    st.pyplot(plt)

# Function to validate audio (check for NaN or Inf)
def validate_audio(audio):
    if not np.isfinite(audio).all():
        st.warning("Invalid values found in audio (NaN or Inf). Replacing with zeros.")
        audio = np.nan_to_num(audio)
    return audio

# Function to normalize audio safely
def normalize_audio(audio):
    max_value = np.max(np.abs(audio))
    if max_value == 0:
        st.warning("Audio has zero max amplitude. Skipping normalization.")
        return audio
    return audio / max_value

# Function to correct pitch to the nearest note in the C Minor scale
def correct_pitch(pitch, scale):
    if pitch == 0:
        return 0
    nearest_pitch = min(scale, key=lambda x: abs(x - pitch))
    return nearest_pitch

# Function to apply pitch correction with intensity control
def apply_pitch_correction(audio, sample_rate, correction_intensity, blend_factor=0.5, scale=C_MINOR_SCALE):
    # Step 1: Denoise the audio
    st.info("Denoising the audio...")
    audio_denoised = nr.reduce_noise(y=audio, sr=sample_rate)
    audio_denoised = validate_audio(audio_denoised)

    # Step 2: Detect pitches
    pitches, magnitudes = librosa.piptrack(y=audio_denoised, sr=sample_rate)
    detected_pitches = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            detected_pitches.append(pitch)
        else:
            detected_pitches.append(0)

    # Step 3: Correct pitches with intensity control
    corrected_audio = np.copy(audio_denoised)
    for i, pitch in enumerate(detected_pitches):
        if pitch > 0:
            nearest_pitch = correct_pitch(pitch, scale)
            # Calculate pitch shift in semitones
            n_steps = correction_intensity * 12 * np.log2(nearest_pitch / pitch)
            corrected_audio = pyrb.pitch_shift(corrected_audio, sample_rate, n_steps)
            corrected_audio = validate_audio(corrected_audio)

    # Step 4: Blend the original and corrected audio
    final_audio = blend_factor * corrected_audio + (1 - blend_factor) * audio_denoised
    final_audio = validate_audio(final_audio)
    final_audio = normalize_audio(final_audio)
    final_audio = np.clip(final_audio, -1.0, 1.0)

    return final_audio


# Streamlit UI
st.title("AI-Powered Auto-Tuner with Intensity Control for C Minor Scale")

# File upload
uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file is not None:
    # Load the audio file
    audio, sample_rate = librosa.load(uploaded_file, sr=None)
    st.audio(uploaded_file, format="audio/wav")

    # Visualize original audio
    st.write("### Original Audio Waveform")
    visualize_audio(audio, sample_rate, "Original Audio")

    # User inputs for pitch correction intensity
    correction_intensity = st.slider("Correction Intensity", 0.0, 1.0, 0.5)
    blend_factor = st.slider("Blend Factor (Naturalness)", 0.0, 1.0, 0.5)

    # Apply pitch correction
    if st.button("Apply Pitch Correction"):
        with st.spinner("Processing..."):
            corrected_audio = apply_pitch_correction(audio, sample_rate, correction_intensity, blend_factor)

            # Save corrected audio to a buffer
            buffer = io.BytesIO()
            sf.write(buffer, corrected_audio, sample_rate, format="WAV")
            buffer.seek(0)

            st.success("Pitch correction applied successfully!")

            # Play the corrected audio
            st.audio(buffer, format="audio/wav")

            # Visualize corrected audio
            st.write("### Corrected Audio Waveform")
            visualize_audio(corrected_audio, sample_rate, "Corrected Audio")

            # Download button for corrected audio
            st.download_button(
                label="Download Corrected Audio",
                data=buffer,
                file_name="corrected_audio.wav",
                mime="audio/wav"
            )
