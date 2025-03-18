import streamlit as st
from openai import OpenAI
import tempfile
import os
from pydub import AudioSegment
import math
from pyannote.audio import Pipeline

# Whisper API setup
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Pyannote pipeline setup (requires Hugging Face token)
HUGGINGFACE_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HUGGINGFACE_TOKEN)

st.title("🎤 Whisper Transcription with Speaker Diarization")

uploaded_file = st.file_uploader("Upload your audio file", type=["mp3", "wav", "webm", "m4a"])

if uploaded_file:
    st.audio(uploaded_file)

    if st.button("Transcribe & Diarize Speakers"):
        with st.spinner("Processing... (this may take several minutes)"):
            with tempfile.TemporaryDirectory() as tmpdir:
                audio_path = os.path.join(tmpdir, "audio_input.mp3")

                # Convert to MP3 for optimization
                audio_segment = AudioSegment.from_file(uploaded_file)
                audio_segment.export(audio_path, format="mp3", bitrate="64k")

                # Step 1: Speaker Diarization with Pyannote.audio
                diarization_result = pipeline(audio_path)

                # Step 2: Transcription with Whisper
                transcript_response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=open(audio_path, "rb"),
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )

                segments = transcript_response.segments

                # Step 3: Merge Diarization + Transcription Results
                diarized_transcript = []
                for segment in segments:
                    seg_start = segment['start']
                    seg_text = segment["text"]

                    # Find corresponding speaker segment
                    speaker_label = "Unknown"
                    for turn, _, speaker in pipeline.itertracks(yield_label=True, audio=audio_path):
                        if turn_overlaps(turn_start=segment.start, turn_end=segment.end, diarization_start=turn.start, diarization_end=turn.end):
                            speaker_label = speaker_label_map(turn.speaker)
                            break

                    diarized_transcript.append(f"{speaker_label}: {segment.text}")

                final_transcript = "\n\n".join(diarized_transcript)

                st.success("Diarization and transcription complete!")
                st.write("### 🎙️ Diarized Transcript")
                st.text_area("Transcript", final_transcript, height=400)

def turn_overlaps(segment_start, segment_end, turn_start, turn_end):
    # Check if transcription segment overlaps diarization segment
    return max(segment_start, turn_start) < min(segment_end, turn_end)

def speaker_label_map(speaker):
    return f"Speaker {speaker+1}"

