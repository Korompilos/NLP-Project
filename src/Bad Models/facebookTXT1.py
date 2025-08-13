import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

def load_summarizer():
    print("Loading summarization pipeline...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer

def split_sentences(text):
    # Split on sentence-ending punctuation, keeping the punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]

def extract_key_sentences(text, top_n=5):
    sentences = split_sentences(text)
    if len(sentences) <= top_n:
        return text  # not enough sentences to filter

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    scores = np.array(tfidf_matrix.sum(axis=1)).ravel()

    # Get top_n sentence indices, preserving original order
    top_indices = scores.argsort()[-top_n:][::-1]
    top_indices = sorted(top_indices)
    extracted = " ".join([sentences[i] for i in top_indices])
    return extracted

def summarize(text, summarizer):
    extracted_text = extract_key_sentences(text, top_n=5)
    return summarizer(extracted_text, max_length=140, min_length=50, do_sample=False)[0]['summary_text']

def main():
    text = """Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in
our lives. Hope you too, to enjoy it as my deepest wishes.
Thank your message to show our words to the doctor, as his next contract checking, to all of us.
I got this message to see the approved message. In fact, I have received the message from the
professor, to show me, this, a couple of days ago. I am very appreciated the full support of the
professor, for our Springer proceedings publication"""
    
    summarizer = load_summarizer()
    result = summarize(text, summarizer)
    print("Summary:", result)

if __name__ == "__main__":
    main()
