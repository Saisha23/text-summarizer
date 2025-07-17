from transformers import pipeline

# Initialize the summarizer
summarizer = pipeline("summarization")

# Read from input.txt
try:
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    if text.strip():
        # Generate summary
        summary = summarizer(text, max_length=50, min_length=25, do_sample=False)

        # Save to summary.txt
        with open("summary.txt", "w", encoding="utf-8") as f:
            f.write(summary[0]['summary_text'])

        print("✅ Summary written to summary.txt")

    else:
        print("⚠️ input.txt is empty.")

except FileNotFoundError:
    print("❌ input.txt not found. Please create it with content to summarize.")
