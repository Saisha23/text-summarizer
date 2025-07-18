import streamlit as st
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
from youtube_transcript_api import YouTubeTranscriptApi
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

# Streamlit app layout
st.set_page_config(page_title="AI Text Summarizer", layout="centered")
st.title("📚 AI Text Summarizer")
st.markdown("Summarize from **text**, **file**, or **YouTube video**")

option = st.sidebar.radio("Choose Input Type", ["Text", "File", "YouTube Video"])

@st.cache_resource
def load_model():
    try:
        # Load model with device management to prevent meta tensor issues
        model_name = "facebook/bart-large-cnn"  # More stable alternative
        
        # Initialize tokenizer
        tokenizer = BartTokenizer.from_pretrained(model_name)
        
        # Initialize model properly (avoids meta tensor issues)
        if torch.cuda.is_available():
            device = "cuda"
            model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
        else:
            device = "cpu"
            model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
            
        return pipeline("summarization", 
                      model=model, 
                      tokenizer=tokenizer,
                      device=0 if torch.cuda.is_available() else -1)
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

summarizer = load_model()

def generate_summary(text):
    if not summarizer:
        return "Model not available. Please check the console for errors."
    
    if len(text.split()) < 30:
        return "❗ Text is too short to summarize. Please provide more content."
    
    try:
        # Handle long texts by splitting (prevents index out of range)
        max_chunk_length = 1024  # BART's max input size
        
        if len(text) > max_chunk_length:
            chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
            summaries = []
            for chunk in chunks:
                summary = summarizer(chunk, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
                summaries.append(summary)
            return " ".join(summaries)
        else:
            summary = summarizer(text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
            return summary
    except Exception as e:
        return f"Error generating summary: {str(e)}"

if option == "Text":
    user_input = st.text_area("Enter text to summarize", height=200)
    if st.button("Summarize"):
        if user_input:
            with st.spinner("Generating Summary..."):
                summary = generate_summary(user_input)
                st.success("✅ Summary")
                st.write(summary)
        else:
            st.warning("Please enter some text to summarize")

elif option == "File":
    uploaded_file = st.file_uploader("Upload a text file (.txt)", type=["txt"])
    if uploaded_file is not None:
        try:
            content = uploaded_file.read().decode("utf-8")
            st.text_area("File Content", value=content, height=200, disabled=True)
            if st.button("Summarize"):
                with st.spinner("Generating Summary..."):
                    summary = generate_summary(content)
                    st.success("✅ Summary")
                    st.write(summary)
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

elif option == "YouTube Video":
    yt_url = st.text_input("Paste YouTube Video URL")
    if st.button("Summarize Captions"):
        if yt_url:
            try:
                video_id = yt_url.split("v=")[-1].split("&")[0]  # More robust URL parsing
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
                full_text = " ".join([t['text'] for t in transcript])
                st.text_area("Transcript", value=full_text, height=200, disabled=True)
                with st.spinner("Summarizing Captions..."):
                    summary = generate_summary(full_text)
                    st.success("✅ Summary")
                    st.write(summary)
            except ValueError:
                st.error("❌ Invalid YouTube URL. Please check the format.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
