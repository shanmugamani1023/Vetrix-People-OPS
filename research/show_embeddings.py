"""
Script to show what happens AFTER chunking:
  Step 1 — Convert chunks to vectors (embeddings)
  Step 2 — Build FAISS index
  Step 3 — Search with a query
Run: python show_embeddings.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── Our 3 chunks from the leave policy PDF ────────────────────────────────────

CHUNKS = [
    {
        "id": 1,
        "locator": "p. 1 (header + definitions)",
        "text": (
            "Global Leave Policy Version 2. Annual leave is paid at the employee's normal rate. "
            "Employees are encouraged to take at least ten working days per year. "
            "Sick leave: 12 working days per year."
        ),
    },
    {
        "id": 2,
        "locator": "1 April 2026 (comparison table)",
        "text": (
            "Version 1 vs Version 2 changes: "
            "Sick leave 10 days → 12 days. "
            "Parental leave primary 16 weeks → 20 weeks. "
            "Bereavement leave 3 days → 5 days."
        ),
    },
    {
        "id": 3,
        "locator": "1 April 2026 (all rules)",
        "text": (
            "Employees may carry forward up to 10 days of unused annual leave. "
            "Sick leave: medical certificate required after 3 consecutive working days. "
            "Parental leave: primary caregiver 20 weeks, secondary caregiver 4 weeks. "
            "Bereavement leave: 5 working days for immediate family. "
            "All leave must be recorded in the HR system."
        ),
    },
]

print("=" * 60)
print("AFTER CHUNKING — What happens next")
print("=" * 60)

# ── STEP 1: Convert text → numbers (embeddings) ───────────────────────────────

print("\nSTEP 1 — Loading embedding model (all-MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("  Model loaded.\n")

texts = [c["text"] for c in CHUNKS]
vectors = model.encode(texts, normalize_embeddings=True)

print(f"  Each chunk → a vector of {vectors.shape[1]} numbers")
print(f"  {len(CHUNKS)} chunks → matrix of shape {vectors.shape}\n")

print("  Example — Chunk 1 first 10 numbers:")
print(f"  {vectors[0][:10].round(4).tolist()}")
print(f"  ...and {vectors.shape[1] - 10} more numbers\n")

print("  These numbers capture MEANING, not exact words.")
print("  Similar meaning → similar numbers → found together in search.\n")

# ── STEP 2: Build FAISS index ─────────────────────────────────────────────────

print("STEP 2 — Building FAISS search index...")
dim = vectors.shape[1]
index = faiss.IndexFlatIP(dim)   # IP = Inner Product (cosine similarity)
index.add(vectors.astype("float32"))
print(f"  Index built. {index.ntotal} vectors stored.\n")

# ── STEP 3: Search with queries ───────────────────────────────────────────────

queries = [
    "How many sick days do I get?",
    "Can I carry forward unused leave?",
    "What is parental leave for new parents?",
]

print("STEP 3 — Searching with user questions\n")
print("-" * 60)

for query in queries:
    q_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(q_vec, k=3)  # top 3 results

    print(f"Query: \"{query}\"")
    print(f"Results (ranked by similarity):")
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
        print(f"  #{rank} Score={score:.3f} | Chunk {CHUNKS[idx]['id']} | {CHUNKS[idx]['locator']}")
    print()

# ── Show similarity between chunks ────────────────────────────────────────────

print("-" * 60)
print("Similarity between chunks (1.0 = identical, 0.0 = unrelated):\n")
for i in range(len(CHUNKS)):
    for j in range(i + 1, len(CHUNKS)):
        sim = float(np.dot(vectors[i], vectors[j]))
        print(f"  Chunk {i+1} vs Chunk {j+1}: {sim:.3f}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
Text chunks  →  [all-MiniLM-L6-v2]  →  384-number vectors
Vectors      →  [FAISS IndexFlatIP]  →  search index

When user asks a question:
  Question → same model → question vector
  FAISS finds closest chunk vectors → top-k results
  Only those chunks are sent to the AI (not all 63!)
""")
