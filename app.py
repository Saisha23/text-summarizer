import os
from transformers import pipeline

# Load summarization pipeline
summarizer = pipeline("summarization")

# List all .txt files in current folder
files = [f for f in os.listdir() if f.endswith(".txt") and not f.startswith("summary_")]

if not files:
    print("📁 No .txt files found to summarize.")
    exit()

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        print(f"⚠️ Skipping empty file: {file}")
        continue

    # Generate summary
    summary = summarizer(text, max_length=50, min_length=25, do_sample=False)

    # Output file name
    out_file = f"summary_{file}"

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(summary[0]['summary_text'])

    print(f"✅ {file} → {out_file}")
