from sentence_transformers import SentenceTransformer, util
import pandas as pd

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sentences with labels
sentences = [
    ("original", "Today is our dragon boat festival, in our Chinese culture , to celebrate it with all safe and great in our lives."),
    ("custom", "It is the day of our dragon boat festival, as part of our Chinese tradition , to celebrate it while making sure all aspects of our lives are safe and great."),
    ("vennify", "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives."),
    ("sshleifer", "Chinese culture celebrates dragon boat festival with all safe and great in our Chinese lives."),
    ("ateeqq", "Our dragon boat festival (in Chinese culture) celebrates today with all safety and great in our lives."),
    ("Chat-GPT", "Today is the Dragon Boat Festival, a cherished tradition in Chinese culture, celebrated with wishes for safety and prosperity in our lives.") # Ground Truth - The best possible model to use
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
