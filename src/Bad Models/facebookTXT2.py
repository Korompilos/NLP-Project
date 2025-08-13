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
    text = """During our final discuss, I told him about the new submission — the one we were waiting since
last autumn, but the updates was confusing as it not included the full feedback from reviewer or
maybe editor?
Anyway, I believe the team, although bit delay and less communication at recent days, they really
tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance
and efforts until the Springer link came finally last week, I think.
Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before
he sending again. Because I didn’t see that part final yet, or maybe I missed, I apologize if so.
Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future
targets"""
    
    summarizer = load_summarizer()
    result = summarize(text, summarizer)
    print("Summary:", result)

if __name__ == "__main__":
    main()
