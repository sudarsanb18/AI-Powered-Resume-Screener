from pathlib import Path
import shutil

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings


# =========================================================
# CONFIGURATION
# =========================================================

CHROMA_DIR = Path("data/chroma_db")
COLLECTION_NAME = "resume_collection"


# =========================================================
# EMBEDDING MODEL
# =========================================================

def get_embedding_model():
    """
    Same embedding model must be used for
    indexing and retrieval.
    """

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# =========================================================
# GET CHROMA CLIENT
# =========================================================

def get_chroma_client():

    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    return chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )


# =========================================================
# GET COLLECTION
# =========================================================

def get_chroma_collection():

    client = get_chroma_client()

    return client.get_or_create_collection(
        name=COLLECTION_NAME
    )


# =========================================================
# CLEAR OLD VECTOR DATABASE
# =========================================================

def clear_vectorstore():

    """
    Completely remove the old ChromaDB.

    IMPORTANT:
    This is called before processing a new batch
    of uploaded resumes.

    Therefore old candidates can NEVER leak into
    the new RAG session.
    """

    print("\n[RESET] Removing old vector database...")

    # First try deleting collection normally
    try:

        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        client.delete_collection(
            name=COLLECTION_NAME
        )

        print("Old Chroma collection deleted ✅")

    except Exception as error:

        print(
            f"Collection deletion skipped: {error}"
        )

    # Remove complete Chroma directory
    if CHROMA_DIR.exists():

        try:

            shutil.rmtree(
                CHROMA_DIR
            )

            print(
                "Old ChromaDB directory removed ✅"
            )

        except Exception as error:

            print(
                f"Warning: Could not remove ChromaDB directory: {error}"
            )

    # Recreate clean directory
    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "Fresh ChromaDB directory created ✅"
    )


# =========================================================
# CREATE FRESH COLLECTION
# =========================================================

def create_fresh_collection():

    """
    Always creates a completely fresh collection.
    """

    clear_vectorstore()

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    print(
        "Fresh resume collection created ✅"
    )

    return collection


# =========================================================
# ADD DOCUMENTS
# =========================================================

def add_documents_to_vectorstore(
    chunks: list[dict],
    collection=None,
):
    """
    Convert chunks into embeddings and store them.
    """

    if not chunks:

        raise ValueError(
            "No chunks available for vectorstore."
        )

    if collection is None:

        collection = get_chroma_collection()

    embedding_model = get_embedding_model()

    texts = []
    metadatas = []
    ids = []

    for index, chunk in enumerate(chunks):

        text = chunk.get(
            "text",
            ""
        ).strip()

        if not text:
            continue

        metadata = chunk.get(
            "metadata",
            {}
        ).copy()

        # -------------------------------------------------
        # Candidate
        # -------------------------------------------------

        source = str(
            metadata.get(
                "source",
                "unknown"
            )
        )

        candidate = str(
            metadata.get(
                "candidate",
                source
            )
        )

        # -------------------------------------------------
        # Page
        # -------------------------------------------------

        page = metadata.get(
            "page",
            0
        )

        # -------------------------------------------------
        # Chunk
        # -------------------------------------------------

        chunk_index = metadata.get(
            "chunk",
            index
        )

        # -------------------------------------------------
        # Normalize metadata
        # -------------------------------------------------

        metadata["source"] = source
        metadata["candidate"] = candidate
        metadata["page"] = page
        metadata["chunk"] = chunk_index

        # -------------------------------------------------
        # Unique ID
        # -------------------------------------------------

        document_id = (
            f"{candidate}"
            f"__page_{page}"
            f"__chunk_{chunk_index}"
        )

        texts.append(text)

        metadatas.append(
            metadata
        )

        ids.append(
            document_id
        )

    if not texts:

        raise ValueError(
            "No valid text chunks found."
        )

    print(
        f"Generating embeddings for {len(texts)} chunks..."
    )

    embeddings = embedding_model.embed_documents(
        texts
    )

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(
        f"Stored {len(texts)} chunks ✅"
    )

    return collection


# =========================================================
# CREATE VECTORSTORE
# =========================================================

def create_vectorstore(
    chunks: list[dict],
    reset: bool = True,
):
    """
    Create vectorstore from chunks.

    reset=True:
        Remove ALL old candidates before indexing.

    reset=False:
        Add to existing collection.
    """

    if not chunks:

        raise ValueError(
            "Cannot create vectorstore without chunks."
        )

    # -----------------------------------------------------
    # IMPORTANT
    # -----------------------------------------------------

    if reset:

        collection = create_fresh_collection()

    else:

        collection = get_chroma_collection()

    # -----------------------------------------------------
    # Add documents
    # -----------------------------------------------------

    collection = add_documents_to_vectorstore(
        chunks=chunks,
        collection=collection,
    )

    return collection


# =========================================================
# VERIFY
# =========================================================

def verify_vectorstore():

    collection = get_chroma_collection()

    count = collection.count()

    print(
        f"Documents in ChromaDB: {count}"
    )

    return count


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print(
        "\n" + "=" * 70
    )

    print(
        "VECTORSTORE RESET TEST"
    )

    print(
        "=" * 70
    )

    try:

        collection = create_fresh_collection()

        print(
            "\nFresh vectorstore ready ✅"
        )

        print(
            f"Documents: {collection.count()}"
        )

        print(
            "\n" + "=" * 70
        )

        print(
            "VECTORSTORE RESET SUCCESSFUL ✅"
        )

        print(
            "=" * 70
        )

    except Exception as error:

        print(
            "\n❌ ERROR"
        )

        print(
            "-" * 70
        )

        print(error)