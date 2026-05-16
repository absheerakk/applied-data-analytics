from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def rank_documents(docs, query, top_k=None):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(docs + [query])

    doc_vectors = tfidf_matrix[:-1]
    query_vector = tfidf_matrix[-1]

    scores = cosine_similarity(query_vector, doc_vectors).flatten()
    ranked_idx = np.argsort(-scores)

    if top_k is not None:
        ranked_idx = ranked_idx[:top_k]

    ranking = [
        (rank + 1, float(scores[i]), docs[i])
        for rank, i in enumerate(ranked_idx)
    ]
    return ranking

if __name__ == "__main__":
    corpus = [
        "Data science combines statistics and computer science.",
        "Machine learning is a branch of artificial intelligence.",
        "Deep learning uses neural networks to model complex patterns.",
        "Graph convolutional networks excel at relational data.",
        "Python is the language of choice for data science."
    ]

    user_query = "How is machine learning related to data science?"

    results = rank_documents(corpus, user_query)

    print(f"\nQuery ➜ {user_query}\n")
    print("Rank | Cosine‑Sim | Document")
    print("-" * 60)
    for r, score, doc in results:
        print(f"{r:>4} | {score:>10.4f} | {doc}")