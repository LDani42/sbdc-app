import streamlit as st
import assemblyai as aai
import tempfile
import os
from pydub import AudioSegment

# Set your AssemblyAI API Key securely from Streamlit Secrets
aai.settings.api_key = st.secrets["ASSEMBLYAI_API_KEY"]

st.title("🎤 Audio Transcription with Speaker Diarization")

uploaded_file = st.file_uploader("Upload your audio file", type=["mp3", "wav", "webm", "m4a"])

if uploaded_file:
    st.audio(uploaded_file)

    if st.button("Start Transcription and Speaker Diarization"):
        with st.spinner("Processing transcription & diarization..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                # Convert uploaded audio to optimized MP3
                audio = AudioSegment.from_file(uploaded_file)
                audio.export(tmp_file.name, format="mp3", bitrate="64k")

            # Setup AssemblyAI config with speaker labels
            transcriber = aai.Transcriber()
            config = aai.TranscriptionConfig(speaker_labels=True)

            try:
                transcript = transcriber.transcribe(tmp_file.name, config=config)

                st.success("✅ Transcription complete!")
                transcript_with_speakers = ""
                for utterance in transcript.utterances:
                    transcript_with_speakers += f"Speaker {utterance.speaker}: {utterance.text}\n\n"

                st.download_button(
                    "Download Transcript",
                    data=transcript_with_speakers,
                    file_name="transcript_with_speakers.txt",
                    mime="text/plain"
                )

                st.write("### 🗣️ Transcript with Speaker Labels:")
                st.text_area("Transcript", transcript_with_speakers, height=400)

            except Exception as e:
                st.error(f"Error during transcription: {e}")

            finally:
                os.unlink(tmp_file.name)
