from flask import Flask, render_template, request
import torch
import sentencepiece  # Ensure it's imported
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

app = Flask(__name__)

# Model name for Pegasus
model_name = "google/pegasus-xsum"

# Initialize the tokenizer and model outside try-except block
tokenizer = None
model = None

# Try loading the model and tokenizer
try:
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")

# Set device to GPU if available, otherwise fallback to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
if model:
    model = model.to(device)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/text-summarization', methods=["POST"])
def summarize():
    summary = ""  # Initialize summary as an empty string

    if request.method == "POST":
        # Get the input text from the form
        inputtext = request.form.get("inputtext_", "")

        if inputtext:  # Ensure inputtext is not empty
            input_text = "summarize: " + inputtext

            # Tokenize the input text
            tokenized_text = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True).to(device)

            # Generate summary using the model
            summary_ = model.generate(tokenized_text, min_length=30, max_length=300)

            # Decode the summary
            summary = tokenizer.decode(summary_[0], skip_special_tokens=True)
        else:
            summary = "Please enter some text to summarize."

    return render_template("output.html", data={"summary": summary})

if __name__ == '__main__':
    app.run()


