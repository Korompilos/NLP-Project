from sentence_transformers import SentenceTransformer, util
import pandas as pd

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sentences with labels
sentences = [
    ("original", "I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation."),
    ("custom", "It is clear the team, even though there have been some delays and reduced communication recently, they genuinely gave their best effort on the manuscript and collaboration."),
    ("vennify", " I believe the team, although there was a bit delay and less communication at recent days, really tried best for paper and cooperation."),
    ("sshleifer", "I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation."),
    ("ateeqq", "Despite the fact that bit delay and less communication at recent days, they really tried best for paper and cooperation."),
    ("Chat-GPT", "I believe that although the team has been a bit delayed and communicated less in recent days, they still did their best on the paper and in working together.") # Ground Truth - The best possible model to use
]

# Separate labels and text
labels, texts = zip(*sentences)

# Encode all at once
embeddings = model.encode(texts, convert_to_tensor=True)

# Compute cosine similarity matrix
similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)

# Convert to pandas DataFrame for nicer labeling
df = pd.DataFrame(similarity_matrix.cpu().numpy(), index=labels, columns=labels)

# Print nicely
print("Semantic similarity matrix:")
print(df.round(2))
