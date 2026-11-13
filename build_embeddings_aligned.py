
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
# 2. Load text-ready dataset
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(input_path)

print("Original dataset shape:", df.shape)


# --------------------------------------------------
# 3. Clean and filter usable text
# --------------------------------------------------

df["clean_text"] = (
    df["clean_text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df = df[df["clean_text"] != ""].copy()

df = df.reset_index(drop=True)

print("Usable text rows:", len(df))


# --------------------------------------------------
# 4. Validate required columns
# --------------------------------------------------

required_columns = [
    "username",
    "type",
    "clean_text",
    "engagement_rate",
    "followersatposting",
    "likes",
    "comments",
    "postcreateddate",
    "postcreatedtime",
    "link"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("Required columns verified.")


# --------------------------------------------------
# 5. Load Sentence Transformer model
# --------------------------------------------------

print("\nLoading Sentence Transformer model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded successfully!")


# --------------------------------------------------
# 6. Generate embeddings
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
# 7. Save embeddings
# --------------------------------------------------

np.save(embedding_path, embeddings)

print("\nEmbeddings saved at:")
print(embedding_path)


# --------------------------------------------------
# 8. Prepare aligned metadata
# --------------------------------------------------

metadata_columns = [
    "username",
    "type",
    "clean_text",
    "engagement_rate",
    "followersatposting",
    "likes",
    "comments",
    "postcreateddate",
    "postcreatedtime",
    "link"
]

metadata = df[metadata_columns].copy()

# Stable row identifier for this embedding dataset.
# This identifier refers to the row order in the
# regenerated metadata and embedding matrix.
metadata.insert(
    0,
    "embedding_row_id",
    np.arange(len(metadata))
)

metadata.to_csv(metadata_path, index=False)

print("\nMetadata saved at:")
print(metadata_path)


# --------------------------------------------------
# 9. Final verification
# --------------------------------------------------

saved_embeddings = np.load(embedding_path)

saved_metadata = pd.read_csv(metadata_path)

print("\nVerification completed!")

print("Saved embeddings shape:", saved_embeddings.shape)

print("Saved metadata shape:", saved_metadata.shape)

print("Number of embeddings:", len(saved_embeddings))

print("Number of metadata rows:", len(saved_metadata))

print(
    "Metadata columns:",
    saved_metadata.columns.tolist()
)


if len(saved_embeddings) != len(saved_metadata):
    raise ValueError(
        "ERROR: Embeddings and metadata row counts do not match."
    )

if not np.array_equal(
    saved_metadata["embedding_row_id"].to_numpy(),
    np.arange(len(saved_metadata))
):
    raise ValueError(
        "ERROR: Embedding row IDs are not sequential."
    )

print(
    "\nSUCCESS: Embeddings and metadata are aligned!"
)

print(
    "Date columns and link have been preserved."
)