import streamlit as st
from openai import OpenAI
import tempfile

# Load OpenAI API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("Whisper Audio Transcription")

uploaded_file = st.file_uploader("Upload audio for transcription:", type=["webm", "wav", "mp3", "m4a"])

if uploaded_file:
    st.audio(uploaded_file)

    if st.button("Transcribe Audio"):
        with st.spinner("Transcribing audio..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())  # correct method: getbuffer()
                tmp_file.flush()

                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=open(tmp_file.name, "rb")
                )

            st.success("Transcription complete!")
            st.write("**Transcript:**")
            st.write(transcript.text)
