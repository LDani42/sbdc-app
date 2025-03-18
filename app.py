import streamlit as st
import assemblyai as aai
import tempfile
import os
from pydub import AudioSegment

# Set up AssemblyAI with your API key from Streamlit secrets
aai.settings.api_key = st.secrets["ASSEMBLYAI_API_KEY"]

st.title("🎤 Audio Transcription with Speaker Diarization")

uploaded_file = st.file_uploader("Upload audio file:", type=["mp3", "wav", "webm", "m4a"])

if uploaded_file := uploaded_file := st.file_uploader("Upload audio:", type=["mp3", "wav", "m4a", "webm"]):
    st.audio(uploaded_file)

    if st.button("Start Transcription with Speaker Detection"):
        with st.spinner("Uploading and processing with AssemblyAI..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                audio = AudioSegment.from_file(uploaded_file)
                audio.export(tmp_file.name, format="mp3")

                # Configure AssemblyAI
                aai.settings.api_key = st.secrets["ASSEMBLYAI_API_KEY"]

                transcriber = aai.Transcriber()
                config = aai.TranscriptionConfig(speaker_labels=True)

                try:
                    transcript = transcriber.transcribe(tmp_file.name, config=config)

                    st.success("Transcription & Speaker detection complete!")
                    transcript_with_speakers = ""
                    for utterance in transcript.utterances:
                        transcript_with_speakers += f"Speaker {utterance.speaker}: {utterance.text}\n\n"

                    st.write("**Transcript with Speaker labels:**")
                    st.write(transcript_with_speakers)

                except Exception as e:
                    st.error(f"Error: {e}")

                finally:
                    os.unlink(tmp_file.name)
