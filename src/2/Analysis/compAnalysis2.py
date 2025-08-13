import numpy as np
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine, euclidean
from transformers import AutoTokenizer, AutoModel
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download NLTK data (run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# =============================
# 1. PREPROCESSING
# =============================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    # Lowercase, tokenize by words (simple regex tokenizer)
    tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return tokens

# =============================
# 2. EMBEDDING METHODS (UPDATED)
# =============================
def embed_bert(texts, model_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings

def embed_avg_word2vec(texts, model):
    vectors = []
    for text in texts:
        tokens = preprocess(text)
        word_vecs = [model[w] for w in tokens if w in model]
        if word_vecs:
            vectors.append(np.mean(word_vecs, axis=0))
        else:
            vectors.append(np.zeros(model.vector_size))
    return np.array(vectors)

def load_glove_model(glove_file):
    glove_model = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array(parts[1:], dtype='float32')
            glove_model[word] = vector
    # Create KeyedVectors and add vectors
    vector_size = len(next(iter(glove_model.values())))
    kv = KeyedVectors(vector_size=vector_size)
    kv.add_vectors(list(glove_model.keys()), list(glove_model.values()))
    return kv

def embed_avg_fasttext(texts, ft_model):
    vectors = []
    for text in texts:
        tokens = preprocess(text)
        word_vecs = [ft_model.get_word_vector(w) for w in tokens]
        if word_vecs:
            vectors.append(np.mean(word_vecs, axis=0))
        else:
            vectors.append(np.zeros(ft_model.get_dimension()))
    return np.array(vectors)

# =============================
# 3. SIMILARITY METRICS
# =============================
def similarity_metrics(emb1, emb2):
    return {
        "cosine": 1 - cosine(emb1, emb2),
        "euclidean": euclidean(emb1, emb2)
    }

# =============================
# 4. VISUALIZATION WITH ARROWS
# =============================
def visualize_with_arrows(embeddings, labels, method="PCA"):
    if method == "PCA":
        reducer = PCA(n_components=2)
    else:
        perplexity = min(len(embeddings) - 1, 5)
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    # Original
    plt.scatter(reduced[0, 0], reduced[0, 1], color="red", label="Original", s=150)

    for i in range(1, len(reduced)):
        plt.scatter(reduced[i, 0], reduced[i, 1], label=labels[i])
        plt.arrow(reduced[0, 0], reduced[0, 1],
                  reduced[i, 0] - reduced[0, 0],
                  reduced[i, 1] - reduced[0, 1],
                  color='gray', alpha=0.5, head_width=0.05, length_includes_head=True)

    plt.legend()
    plt.title(f"Embedding Shift Visualization ({method})")
    plt.show()

# =============================
# 5. MAIN ANALYSIS (UPDATED)
# =============================
if __name__ == "__main__":
    original_text = "I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation."
    reconstructed_texts = [
        "I believe the team, although there was a bit delay and less communication at recent days, really tried best for paper and cooperation.",
        "The team has encountered a few setbacks and communication challenges lately, yet they have genuinely tried their best for the paper and teamwork.",
        "Despite the fact that bit delay and less communication at recent days, they really tried best for paper and cooperation.",
        "I believe that although the team has been a bit delayed and communicated less in recent days, they still did their best on the paper and in working together."
    ]
    labels = ["Original", "Vennify", "Ateeqq", "Sshleifer","Chat-GPT"]

    # ====== BERT ======
    print("\n=== BERT ANALYSIS ===")
    texts_all = [original_text] + reconstructed_texts
    emb_bert = embed_bert(texts_all)
    for i in range(1, len(emb_bert)):
        metrics = similarity_metrics(emb_bert[0], emb_bert[i])
        print(f"{labels[i]}: Cosine={metrics['cosine']:.4f}, Euclidean={metrics['euclidean']:.4f}")
    visualize_with_arrows(emb_bert, labels, method="PCA")
    visualize_with_arrows(emb_bert, labels, method="t-SNE")

    # ====== GloVe ======
    print("\n=== GloVe ANALYSIS ===")
    glove_model = load_glove_model("glove.6B.300d.txt")
    emb_glove = embed_avg_word2vec(texts_all, glove_model)
    for i in range(1, len(emb_glove)):
        metrics = similarity_metrics(emb_glove[0], emb_glove[i])
        print(f"{labels[i]}: Cosine={metrics['cosine']:.4f}, Euclidean={metrics['euclidean']:.4f}")
    visualize_with_arrows(emb_glove, labels, method="PCA")
    visualize_with_arrows(emb_glove, labels, method="t-SNE")

    # ====== FastText ======
    print("\n=== FastText ANALYSIS ===")
    import fasttext
    import fasttext.util
    fasttext.util.download_model('en', if_exists='ignore')
    ft = fasttext.load_model('cc.en.300.bin')
    emb_fasttext = embed_avg_fasttext(texts_all, ft)
    for i in range(1, len(emb_fasttext)):
        metrics = similarity_metrics(emb_fasttext[0], emb_fasttext[i])
        print(f"{labels[i]}: Cosine={metrics['cosine']:.4f}, Euclidean={metrics['euclidean']:.4f}")
    visualize_with_arrows(emb_fasttext, labels, method="PCA")
    visualize_with_arrows(emb_fasttext, labels, method="t-SNE")
