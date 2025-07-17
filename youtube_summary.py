import sys
import re
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline

# Load summarization pipeline
summarizer = pipeline("summarization")

def get_video_id(url):
    """Extract video ID from YouTube URL"""
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None

if len(sys.argv) != 2:
    print("Usage: python youtube_summary.py <YouTube_URL>")
    exit()

video_url = sys.argv[1]
video_id = get_video_id(video_url)

if not video_id:
    print("❌ Invalid YouTube URL.")
    exit()

try:
    # Get transcript (in English)
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
    text = " ".join([entry['text'] for entry in transcript])

    # Summarize in chunks (YouTube transcripts can be long)
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    summaries = [summarizer(chunk, max_length=50, min_length=25, do_sample=False)[0]['summary_text'] for chunk in chunks]

    full_summary = "\n".join(summaries)

    with open(f"summary_{video_id}.txt", "w", encoding="utf-8") as f:
        f.write(full_summary)

    print(f"✅ Summary saved to summary_{video_id}.txt")

except Exception as e:
    print("❌ Error fetching transcript:", e)
