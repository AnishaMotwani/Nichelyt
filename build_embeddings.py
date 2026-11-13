
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path


# --------------------------------------------------
# 1. Define file paths
# --------------------------------------------------

input_path = Path("data/processed/dataset2_text_ready.csv")

embedding_path = Path("data/processed/caption_embeddings.npy")

metadata_path = Path("data/processed/retrieval_metadata.csv")


# --------------------------------------------------
# 2. Load the text-ready dataset
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(input_path)

print("Original dataset shape:", df.shape)


# --------------------------------------------------
# 3. Keep only usable text
# --------------------------------------------------

df["clean_text"] = df["clean_text"].fillna("").astype(str).str.strip()

df = df[df["clean_text"] != ""].copy()

df = df.reset_index(drop=True)

print("Usable text rows:", len(df))


# --------------------------------------------------
# 4. Load the Sentence Transformer model
# --------------------------------------------------

print("\nLoading Sentence Transformer model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded successfully!")


# --------------------------------------------------
# 5. Generate embeddings
# --------------------------------------------------

print("\nGenerating embeddings...")

embeddings = model.encode(
    df["clean_text"].tolist(),
    normalize_embeddings=True,
    show_progress_bar=True
)

print("Embeddings generated successfully!")

print("Embedding matrix shape:", embeddings.shape)


# --------------------------------------------------
# 6. Save embeddings
# --------------------------------------------------

np.save(embedding_path, embeddings)

print("\nEmbeddings saved at:")
print(embedding_path)


# --------------------------------------------------
# 7. Save matching metadata
# --------------------------------------------------

metadata_columns = [
    "username",
    "type",
    "clean_text",
    "engagement_rate",
    "followersatposting",
    "likes",
    "comments"
]

available_columns = [
    column for column in metadata_columns
    if column in df.columns
]

metadata = df[available_columns].copy()

metadata.to_csv(metadata_path, index=False)

print("\nMetadata saved at:")
print(metadata_path)


# --------------------------------------------------
# 8. Final verification
# --------------------------------------------------

saved_embeddings = np.load(embedding_path)

saved_metadata = pd.read_csv(metadata_path)

print("\nVerification completed!")

print("Saved embeddings shape:", saved_embeddings.shape)

print("Saved metadata shape:", saved_metadata.shape)

print("Number of embeddings:", len(saved_embeddings))

print("Number of metadata rows:", len(saved_metadata))

if len(saved_embeddings) == len(saved_metadata):
    print("\nSUCCESS: Embeddings and metadata are aligned!")
else:
    print("\nWARNING: Embeddings and metadata row counts do not match!")