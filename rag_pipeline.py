import math
import re
from collections import Counter

# ==========================================================
# Module 04: RAG, Embeddings & Vector Databases
# Project: Chat With Your Documents
# Pipeline: Document -> Chunking -> Embeddings -> Vector DB -> Retrieval -> LLM -> Answer
# ==========================================================

# 1. Sample Knowledge Base Documents
DOCUMENTS = [
    {
        "source": "internship_guidelines.txt",
        "text": (
            "Codomax Virtual Internship 2026 runs across 6 structured modules. "
            "Module 03 covers LLM APIs and chat history management. "
            "Module 04 focuses on Retrieval-Augmented Generation (RAG), embeddings, "
            "text chunking, and vector databases to reduce LLM hallucinations."
        ),
    },
    {
        "source": "rag_architecture_notes.txt",
        "text": (
            "Retrieval-Augmented Generation splits large documents into smaller chunks. "
            "Each chunk is converted into a numerical embedding vector and stored in a vector database. "
            "When a user asks a question, the query is embedded and matched using cosine similarity "
            "to retrieve the most relevant context before generating an answer."
        ),
    },
]


# 2. Text Chunking (with overlap)
def chunk_text(text: str, chunk_size: int = 25, overlap: int = 5) -> list[str]:
    words = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


# 3. Lightweight Text Embedding & Cosine Similarity
def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def embed_text(text: str, vocab: list[str]) -> list[float]:
    counts = Counter(tokenize(text))
    return [float(counts[word]) for word in vocab]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# 4. In-Memory Vector Database
class SimpleVectorDB:
    def __init__(self):
        self.records = []
        self.vocab = []

    def index_documents(self, docs: list[dict]):
        all_chunks = []
        for doc in docs:
            for chunk in chunk_text(doc["text"]):
                all_chunks.append({"source": doc["source"], "chunk": chunk})

        vocab_set = set()
        for item in all_chunks:
            vocab_set.update(tokenize(item["chunk"]))
        self.vocab = sorted(vocab_set)

        for item in all_chunks:
            self.records.append(
                {
                    "source": item["source"],
                    "chunk": item["chunk"],
                    "embedding": embed_text(item["chunk"], self.vocab),
                }
            )

    def search(self, query: str, top_k: int = 2) -> list[dict]:
        query_vec = embed_text(query, self.vocab)
        scored = []
        for rec in self.records:
            score = cosine_similarity(query_vec, rec["embedding"])
            scored.append({**rec, "score": round(score, 3)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


# 5. Grounded RAG Answer Generator
def ask_documents(db: SimpleVectorDB, question: str) -> str:
    matches = db.search(question, top_k=2)
    if not matches or matches[0]["score"] == 0.0:
        return "I could not find relevant context in the uploaded documents."

    context = "\n".join(
        f"- [{m['source']} | score={m['score']}]: {m['chunk']}" for m in matches
    )
    return (
        f"Question: {question}\n"
        f"Retrieved Context:\n{context}\n"
        f"Grounded Answer: Based on {matches[0]['source']}, {matches[0]['chunk']}"
    )


if __name__ == "__main__":
    vector_db = SimpleVectorDB()
    vector_db.index_documents(DOCUMENTS)

    sample_questions = [
        "How does RAG prevent LLM hallucinations?",
        "How are document chunks matched to a user query?",
    ]
    for q in sample_questions:
        print("=" * 55)
        print(ask_documents(vector_db, q))
