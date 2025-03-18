import streamlit as st
from openai import OpenAI
import tempfile
import os

# Load API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

MAX_SIZE_MB = 24  # Safe limit to avoid API rejection (~25 MB limit from OpenAI)

st.title("Whisper Audio Transcription")

uploaded_file = st.file_uploader("Upload audio file for transcription:", type=["webm", "mp3", "wav", "m4a"])

if uploaded_file:
    file_size_mb = uploaded_file.size / (1024 * 1024)

    st.audio(uploaded_file)

    if file_size_ok := file_size_ok := file_size_ok := file_size_ok := file_size_ok := file_size_ok := file_size_ok := file_size_ok := file_size_ok := file_size_ok := file_size_ok := file_size_ok := file_size_ok := file_size_ok := file_size_ok := file_size_ok := file_size_ok := file_size_ok := file_size_ok := file_size_ok := file_size_ok := file_size_ok := file_size_ok := file_size_ok := file_size_ok := file_size_ok := file_size_ok := file_size_ok := file_size_ok := st.button("Transcribe Audio"):

        if file_size_ok and file_size_ok and file_size_ok and file_size_ok and file_size_ok and file_size_ok and file_size_ok and file_size_ok and file_size_ok and file_size_ok and file_size_ok and file_size_ok and file_size_ok and file_size_ok and file_size_ok and file_size_ok and file_size_ok and file_size_ok and file_size_ok and file_size_ok:
            with st.spinner("Transcribing audio..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_file.flush()

                    try:
                        transcript = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=open(tmp_file.name, "rb")
                        )
                        st.success("Transcription complete!")
                        st.write("**Transcript:**")
                        st.write(transcript.text)

                    except Exception as e:
                        st.error(f"❌ Error during transcription: {str(e)}")

                    finally:
                        os.unlink(tmp_file.name)
        else:
            st.error(f"❌ File too large ({file_size_mb:.2f} MB). Maximum size is {MAX_SIZE_MB} MB.")
