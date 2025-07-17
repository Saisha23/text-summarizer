import sys
from transformers import pipeline

# Check for correct number of arguments
if len(sys.argv) != 3:
    print("❌ Usage: python app.py <input_file> <output_file>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

# Load summarizer
summarizer = pipeline("summarization")

try:
    # Read input file
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    if text.strip():
        # Generate summary
        summary = summarizer(text, max_length=50, min_length=25, do_sample=False)

        # Write summary to output file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(summary[0]['summary_text'])

        print(f"✅ Summary saved to {output_file}")
    else:
        print("⚠️ Input file is empty.")

except FileNotFoundError:
    print(f"❌ File '{input_file}' not found.")
