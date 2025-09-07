from sentence_transformers import SentenceTransformer
import torch

model_name = "sentence-transformers/all-MiniLM-L6-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device selected:", device)

model = SentenceTransformer(model_name, device=device)
vec = model.encode("test", normalize_embeddings=True)
print("Vector length:", len(vec))
