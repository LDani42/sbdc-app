import streamlit as st
from openai import OpenAI
import tempfile
import os
from pydub import AudioSegment
import math

# Whisper setup
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("🎧 Long Audio Transcription (Auto-Split)")

uploaded_file = st.file_uploader("Upload your long audio file", type=["mp3", "wav", "webm", "m4a"])

CHUNK_LENGTH_MS = 10 * 60 * 1000  # 10 min chunks

def transcribe_audio_chunk(audio_chunk_path):
    with open(audio_chunk_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcript.text

if uploaded_file:
    st.audio(uploaded_file)
    audio_size_mb = uploaded_file.size / (1024 * 1024)
    st.write(f"Uploaded audio size: {audio_size_mb:.2f} MB")

    if st.button("Start Transcription"):
        with st.spinner("Splitting audio and transcribing..."):
            full_transcript = ""
            with tempfile.TemporaryDirectory() as tmpdir:
                audio_path = os.path.join(tmpdir, "uploaded_audio")
                with open(audio_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                audio = AudioSegment.from_file(audio_path)
                total_length_ms = len(audio)
                num_chunks = math.ceil(total_length_ms / CHUNK_LENGTH_MS)

                st.write(f"Audio duration: {total_length_ms / 60000:.2f} mins, splitting into {num_chunks} chunks.")

                transcripts = []
                progress_bar = st.progress(0)

                for i in range(num_chunks):
                    start_ms = i * CHUNK_LENGTH_MS
                    end_ms = min((i + 1) * CHUNK_LENGTH_MS, total_length_ms)
                    chunk = audio[start_ms:end_ms]

                    chunk_file = os.path.join(tmpdir, f"chunk_{i}.mp3")
                    chunk.export(chunk_file, format="mp3")

                    st.write(f"Transcribing chunk {i + 1}/{num_chunks}...")
                    try:
                        text = transcribe_audio_chunk(chunk_file)
                        transcripts.append(text)
                    except Exception as e:
                        transcripts.append(f"[Error transcribing chunk {i + 1}: {str(e)}]")
                    progress_bar.progress((i + 1) / num_chunks)

                full_transcript = "\n\n".join(transcripts)

            st.success("✅ Transcription complete!")
            st.download_button(
                "Download Full Transcript",
                data=full_transcript,
                file_name="full_transcript.txt",
                mime="text/plain"
            )

            st.subheader("Transcript:")
            st.write(full_transcript)
