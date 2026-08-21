from pathlib import Path

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings


# =========================================================
# CONFIGURATION
# =========================================================

CHROMA_DIR = Path("data/chroma_db")
COLLECTION_NAME = "resume_collection"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# =========================================================
# EMBEDDING MODEL
# =========================================================

def get_embedding_model():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


# =========================================================
# GET CHROMA COLLECTION
# =========================================================

def get_collection():

    if not CHROMA_DIR.exists():

        raise FileNotFoundError(
            "ChromaDB not found. "
            "Please upload/analyze resumes first."
        )

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    try:

        collection = client.get_collection(
            name=COLLECTION_NAME
        )

    except Exception as error:

        raise FileNotFoundError(
            "Resume ChromaDB collection not found."
        ) from error

    return collection


# =========================================================
# RETRIEVE
# =========================================================

def retrieve(
    query: str,
    top_k: int = 3,
    candidate: str | None = None,
):

    if not query or not query.strip():

        raise ValueError(
            "Query cannot be empty."
        )

    embedding_model = get_embedding_model()

    collection = get_collection()

    query_embedding = embedding_model.embed_query(
        query
    )

    # -----------------------------------------------------
    # Build query
    # -----------------------------------------------------

    query_kwargs = {

        "query_embeddings": [
            query_embedding
        ],

        "n_results": top_k,
    }

    # -----------------------------------------------------
    # IMPORTANT:
    # Search only inside selected candidate
    # -----------------------------------------------------

    if candidate:

        query_kwargs["where"] = {
            "candidate": candidate
        }

    results = collection.query(
        **query_kwargs
    )

    documents = (
        results.get("documents", [[]])[0]
    )

    metadatas = (
        results.get("metadatas", [[]])[0]
    )

    distances = (
        results.get("distances", [[]])[0]
    )

    retrieved_documents = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        retrieved_documents.append({

            "text": document,

            "metadata": metadata,

            "distance": distance

        })

    return retrieved_documents


# =========================================================
# GET ALL CANDIDATES
# =========================================================

def get_candidates():

    collection = get_collection()

    data = collection.get(
        include=["metadatas"]
    )

    candidates = set()

    for metadata in data.get(
        "metadatas",
        []
    ):

        if not metadata:
            continue

        candidate = metadata.get(
            "candidate"
        )

        if candidate:
            candidates.add(candidate)

    return sorted(candidates)


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("RAG RETRIEVER TEST")
    print("=" * 70)

    try:

        candidates = get_candidates()

        print("\nCandidates in database:")

        for candidate in candidates:

            print(
                f"  • {candidate}"
            )

        query = (
            "What experience does this candidate "
            "have in sales and managing teams?"
        )

        print(
            f"\nQuery:\n{query}"
        )

        results = retrieve(
            query=query,
            top_k=3
        )

        print(
            "\nRetrieved chunks:",
            len(results)
        )

        for i, result in enumerate(
            results,
            start=1
        ):

            metadata = result["metadata"]

            print("\n" + "-" * 70)

            print(
                f"RESULT {i}"
            )

            print(
                "Candidate:",
                metadata.get(
                    "candidate",
                    "Unknown"
                )
            )

            print(
                "Page:",
                metadata.get(
                    "page",
                    "Unknown"
                )
            )

            print(
                "\n",
                result["text"]
            )

        print(
            "\n" + "=" * 70
        )

        print(
            "RETRIEVAL SUCCESSFUL ✅"
        )

        print(
            "=" * 70
        )

    except Exception as error:

        print("\n❌ ERROR")
        print("-" * 70)
        print(error)