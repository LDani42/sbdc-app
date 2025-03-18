import streamlit as st
import openai
import tempfile

openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("Whisper API Audio Transcription")

audio_file = st.file_uploader("Upload audio for transcription", type=["webm", "mp3", "wav"])

if audio_file:
    st.audio(audio_file)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        tmp.write(audio_file.getbuffer())
        tmp_path = tmp.name

    with open(tmp_path, "rb") as file:
        transcript = openai.Audio.transcribe("whisper-1", file)

    st.write("**Transcript:**")
    st.success(transcript["text"])
