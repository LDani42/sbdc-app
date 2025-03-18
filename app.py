import streamlit as st
from openai import OpenAI
import tempfile

# Load your OpenAI API key securely from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("🎙️ Whisper Audio Transcription")

uploaded_audio = st.file_uploader("Upload audio file to transcribe", type=["wav", "mp3", "webm", "m4a"])

if uploaded_audio is not None:
    st.audio(uploaded_audio, format='audio/webm')

    if st.button("Transcribe Audio"):
        with st.spinner("Transcribing..."):
            # Save the uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file.flush()

                # Send the file to Whisper API for transcription
                with open(tmp_file.name, "rb") as audio:
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio
                    )

            st.success("Transcription complete!")
            st.write("**Transcript:**")
            st.write(transcript.text)
