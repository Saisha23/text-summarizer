import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import re
import textwrap

# Initialize summarizer
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Extract YouTube video ID
def get_youtube_id(url):
    regex = (
        r"(?:https?:\/\/)?(?:www\.)?"
        r"(?:youtube\.com\/(?:watch\?v=|embed\/|v\/)|youtu\.be\/)"
        r"([\w\-]{11})"
    )
    match = re.search(regex, url)
    return match.group(1) if match else None

# Fetch transcript
def fetch_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    full_text = " ".join([t['text'] for t in transcript])
    return full_text

# Break long text into chunks for summarization
def split_text(text, max_tokens=800):
    wrapper = textwrap.TextWrapper(width=max_tokens)
    return wrapper.wrap(text)

# Streamlit UI
st.title("🎬 YouTube Video Summarizer")

url = st.text_input("📥 Paste YouTube video URL:")

if url:
    try:
        video_id = get_youtube_id(url)
        st.info("Fetching transcript...")
        full_transcript = fetch_transcript(video_id)
        st.success("Transcript fetched successfully ✅")

        st.subheader("🧠 Generating Summary...")

        chunks = split_text(full_transcript, max_tokens=800)
        summary = ""

        for chunk in chunks:
            summary_piece = summarizer(chunk, max_length=130, min_length=10, do_sample=False)[0]['summary_text']
            summary += summary_piece + "\n\n"

        st.text_area("📝 Summary:", summary.strip(), height=300)

    except Exception as e:
        st.error(f"❌ Error: {e}")
