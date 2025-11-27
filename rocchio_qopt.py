import numpy as np
import random

# Fix random seed for reproducibility (optional)
random.seed(42)
np.random.seed(42)

vocabulary = [
    "team", "coach", "hockey", "baseball", "soccer",
    "penalty", "score", "win", "loss", "season"
]

NUM_DOCS = 50
VOCAB_SIZE = len(vocabulary)

# Generate tf-idf-like vectors for 50 documents

def generate_document_vector(vocab_size):
    """
    Generate one document vector:
    For each term:
        - with some probability (e.g. 0.5) the term appears
        - if it appears, assign a random weight between 2 and 6
        - if not, weight is 0
    """
    vec = np.zeros(vocab_size, dtype=float)
    for i in range(vocab_size):
        appears = random.random() < 0.5  # 50% chance
        if appears:
            vec[i] = random.uniform(2.0, 6.0)
        else:
            vec[i] = 0.0
    return vec

# Create all documents

docs = np.array([generate_document_vector(VOCAB_SIZE) for _ in range(NUM_DOCS)])

# Step A: Choose relevant (20%) and non relevant (80%) docs

num_relevant = int(0.2 * NUM_DOCS)  # 20% relevant
all_indices = list(range(NUM_DOCS))
random.shuffle(all_indices)

relevant_indices = all_indices[:num_relevant]
non_relevant_indices = all_indices[num_relevant:]

labels = ["relevant" if i in relevant_indices else "non relevant"
          for i in range(NUM_DOCS)]

#  Compute q_opt = 2 * μ_R - μ_NR

mu_R = docs[relevant_indices].mean(axis=0)
mu_NR = docs[non_relevant_indices].mean(axis=0)
q_opt = 2 * mu_R - mu_NR

#  cosine similarity

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors a and b.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

#  Top 5 features in q_opt

top_k = 5
sorted_indices = np.argsort(q_opt)[::-1]  # descending order
top_feature_indices = sorted_indices[:top_k]

print("===== Top 5 features in q_opt =====")
for idx in top_feature_indices:
    print(f"Feature: {vocabulary[idx]:>8s}, weight in q_opt: {q_opt[idx]:.4f}")
print()

# 3 closest documents to q_opt 

doc_similarities = []

for i in range(NUM_DOCS):
    sim = cosine_similarity(q_opt, docs[i])
    doc_similarities.append((i, sim, labels[i]))

# Sort by similarity 

doc_similarities.sort(key=lambda x: x[1], reverse=True)

print("===== Top 3 documents closest to q_opt =====")
top_docs = doc_similarities[:3]
for idx, sim, label in top_docs:
    vec = docs[idx]
    vec_rounded = [round(float(x), 3) for x in vec]
    print(f"Document num: {idx}")
    print(f"Label       : {label}")
    print(f"Cosine sim  : {sim:.4f}")
    print(f"Vector      : {vec_rounded}")
    print("-" * 50)

# print summary of how many relevant / non relevant

num_rel = sum(1 for lab in labels if lab == "relevant")
num_non_rel = sum(1 for lab in labels if lab == "non relevant")
print()
print("===== summary =====")
print(f"Total documents   : {NUM_DOCS}")
print(f"Relevant documents: {num_rel}")
print(f"Non relevant docs : {num_non_rel}")
