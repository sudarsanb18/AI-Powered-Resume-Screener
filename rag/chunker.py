from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> list[dict]:
    """
    Split extracted PDF documents into smaller chunks.

    Each chunk keeps the original metadata such as:
    - source filename
    - page number
    """

    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = []

    for document in documents:

        text = document.get("text", "").strip()
        metadata = document.get("metadata", {}).copy()

        if not text:
            continue

        split_texts = splitter.split_text(text)

        for chunk_index, chunk_text in enumerate(split_texts):

            chunk_text = chunk_text.strip()

            if not chunk_text:
                continue

            chunk_metadata = metadata.copy()

            chunk_metadata["chunk"] = chunk_index

            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": chunk_metadata,
                }
            )

    return chunks


# ---------------------------------------------------------
# COMPATIBILITY FUNCTION
# ---------------------------------------------------------

def create_chunks(documents: list[dict]) -> list[dict]:
    """
    Compatibility wrapper used by vectorstore.py.

    Internally uses chunk_documents().
    """
    return chunk_documents(documents)


# ---------------------------------------------------------
# TEST CHUNKER
# ---------------------------------------------------------

if __name__ == "__main__":

    from loader import load_pdf

    pdf_path = (
        r"C:\Users\sudar\OneDrive\Documents\P"
        r"\ML-Project\data\resume\GRM.pdf"
    )

    print("\n" + "=" * 60)
    print("RAG CHUNKER TEST")
    print("=" * 60)

    try:

        # Step 1: Load PDF
        documents = load_pdf(pdf_path)

        print(f"\nPages loaded: {len(documents)}")

        # Step 2: Create chunks
        chunks = create_chunks(documents)

        print(f"Chunks created: {len(chunks)}")

        # Step 3: Display chunks
        for index, chunk in enumerate(chunks, start=1):

            print("\n" + "=" * 60)
            print(f"CHUNK {index}")
            print("=" * 60)

            print(
                f"Source : "
                f"{chunk['metadata'].get('source')}"
            )

            print(
                f"Page   : "
                f"{chunk['metadata'].get('page')}"
            )

            print(
                f"Chunk  : "
                f"{chunk['metadata'].get('chunk')}"
            )

            print("\nText:")
            print(chunk["text"])

        print("\n" + "=" * 60)
        print("CHUNKING SUCCESSFUL ✅")
        print("=" * 60)

    except Exception as e:

        print("\n❌ ERROR")
        print("-" * 60)
        print(e)