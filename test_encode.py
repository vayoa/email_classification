import time
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# This will download the model to your machine and set it up for GPU support
ef = SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda"
)

# Test with 10k documents
docs = []
for i in range(10000):
    docs.append(f"this is a document with id {i}")

start_time = time.perf_counter()
embeddings = ef(docs)
end_time = time.perf_counter()
print(f"Elapsed time: {end_time - start_time} seconds")
