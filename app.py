import streamlit as st
from openai import OpenAI
import tempfile

# Load API key from Streamlit Secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("🎙 Whisper Audio Transcription")

uploaded_file = st.file_uploader("Upload an audio file for transcription", type=['mp3', 'mp3', 'wav', 'webm'])

if uploaded_file := st.file_uploader("Upload audio to transcribe:", type=["wav", "mp3", "webm", "m4a"]):
    st.audio(uploaded_file, format='audio/webm')

    if st.button("Transcribe Audio"):
        with st.spinner("Transcribing..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_file:
                tmp_file_name = tmp.name
                tmp_audio = uploaded_file.getvalue()
                tmp_file_path = tmp.name
                with open(tmp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())

                # Transcribe using OpenAI Whisper
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=open(tmp_file.name, "rb")
                )

            st.success("Transcription complete!")
            st.write("**Transcript:**")
            st.write(transcript.text)
