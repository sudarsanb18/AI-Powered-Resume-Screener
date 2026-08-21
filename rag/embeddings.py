from langchain_huggingface import HuggingFaceEmbeddings


# ---------------------------------------------------------
# CREATE EMBEDDING MODEL
# ---------------------------------------------------------

def get_embedding_model():
    """
    Load the local sentence-transformer embedding model.

    This model converts text into numerical vectors
    that can be stored and searched in a vector database.
    """

    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return model


# ---------------------------------------------------------
# TEST EMBEDDINGS
# ---------------------------------------------------------

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("RAG EMBEDDING TEST")
    print("=" * 60)

    try:

        # Load embedding model
        embedding_model = get_embedding_model()

        print("\nEmbedding model loaded successfully ✅")

        # Test text
        test_texts = [
            "Python machine learning developer",
            "Sales manager with 20 years of experience",
            "Experienced in team management and business development",
        ]

        # Generate embeddings
        embeddings = embedding_model.embed_documents(test_texts)

        print(f"\nTexts embedded: {len(embeddings)}")

        # Display vector information
        for index, embedding in enumerate(embeddings, start=1):

            print("\n" + "-" * 60)
            print(f"Text {index}:")
            print(test_texts[index - 1])

            print(f"\nVector dimensions: {len(embedding)}")

            print("\nFirst 10 values:")
            print(embedding[:10])

        print("\n" + "=" * 60)
        print("EMBEDDING GENERATION SUCCESSFUL ✅")
        print("=" * 60)

    except Exception as e:

        print("\n❌ ERROR")
        print("-" * 60)
        print(e)