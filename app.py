import streamlit as st
from openai import OpenAI
import tempfile
import os

# Load your OpenAI API key securely from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

MAX_SIZE_MB = 24  # Whisper API max ~25 MB limit, keeping a buffer

st.title("🎙️ Whisper Audio Transcription")

uploaded_file = st.file_uploader("Upload audio file to transcribe", type=["webm", "wav", "mp3", "m4a"])

if uploaded_file:
    file_size_mb = uploaded_file.size / (1024 * 1024)
    st.audio(uploaded_file)

    if st.button("Transcribe Audio"):
        if file_size_mb > MAX_SIZE_MB:
            st.error(f"❌ File too large ({file_size_mb:.2f} MB). Max size allowed is {MAX_SIZE_MB} MB.")
        else:
            with st.spinner("Transcribing audio..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_file.flush()

                    try:
                        with open(tmp_file.name, "rb") as audio_file:
                            transcript = client.audio.transcriptions.create(
                                model="whisper-1",
                                file=audio_file
                            )
                        st.success("Transcription complete!")
                        st.write("**Transcript:**")
                        st.write(transcript.text)

                    except Exception as e:
                        st.error(f"❌ Error during transcription: {str(e)}")

                    finally:
                        os.unlink(tmp_file.name)
