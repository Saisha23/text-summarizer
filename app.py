from transformers import pipeline

# Create a summarizer pipeline
summarizer = pipeline("summarization")

# Sample input text
text = """
Artificial Intelligence and Data Science is a fast-growing field with applications in every industry. 
From healthcare to finance, AI models are being used to improve efficiency and make data-driven decisions. 
Students in AI & DS learn to use tools like machine learning, natural language processing, and deep learning to build smart applications.
"""

# Generate summary
summary = summarizer(text, max_length=50, min_length=25, do_sample=False)

# Print summary
print("🔍 Summary:")
print(summary[0]['summary_text'])
