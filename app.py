import streamlit as st
from openai import OpenAI
import tempfile
import os
from pydub import AudioSegment
import math

# Whisper API setup
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("🎧 Enhanced Long Audio Transcription")

uploaded_file = st.file_uploader("Upload audio (any format)", type=["mp3", "wav", "webm", "m4a"])

CHUNK_LENGTH_MS = 10 * 60 * 1000  # 10 min chunks

def transcribe_chunk(chunk_path):
    with open(chunk_path, "rb") as audio:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=audio)
    return transcript.text

if uploaded_file:
    st.audio(uploaded_file)

    if st.button("Transcribe (Optimized)"):
        with st.spinner("Optimizing audio & transcribing..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                original_audio_path = os.path.join(tmpdir, "original_audio")
                optimized_audio_path = os.path.join(tmpdir, "optimized_audio.mp3")

                # Save original file
                with open(original_audio_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Convert original audio to MP3 to compress size
                audio = AudioSegment.from_file(original_audio_path)
                audio.export(optimized_audio_path, format="mp3", bitrate="64k")

                # Load optimized audio
                audio_mp3 = AudioSegment.from_mp3(optimized_audio_path)
                total_length_ms = len(audio_mp3)
                num_chunks = math.ceil(total_length_ms / CHUNK_LENGTH_MS)

                st.write(f"Audio duration: {total_length_ms / 60000:.2f} mins, split into {num_chunks} chunks.")

                transcripts = []
                progress_bar = st.progress(0)

                for i in range(num_chunks):
                    start_ms = i * CHUNK_LENGTH_MS
                    end_ms = min((i + 1) * CHUNK_LENGTH_MS, total_length_ms)
                    chunk = audio_mp3[start_ms:end_ms]

                    chunk_path = os.path.join(tmpdir, f"chunk_{i}.mp3")
                    chunk.export(chunk_path, format="mp3", bitrate="64k")

                    st.write(f"Transcribing chunk {i + 1}/{num_chunks}...")
                    try:
                        transcript = transcribe_chunk(chunk_path)
                        transcripts.append(transcript)
                    except Exception as e:
                        transcripts.append(f"[Error chunk {i + 1}: {str(e)}]")

                    progress_bar.progress((i + 1) / num_chunks)

                full_transcript = "\n\n".join(transcripts)

            st.success("✅ Transcription complete!")
            st.download_button(
                "Download Full Transcript",
                data=full_transcript,
                file_name="transcript.txt",
                mime="text/plain"
            )

            st.subheader("Transcript:")
            st.write(full_transcript)
