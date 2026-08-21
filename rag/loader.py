from pathlib import Path
from pypdf import PdfReader


def load_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from a PDF while preserving page-level metadata.
    """

    path = Path(pdf_path)

    # Check whether the file exists
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Check file type
    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    # Read PDF
    reader = PdfReader(str(path))

    documents = []

    # Extract each page
    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""
        text = text.strip()

        # Skip empty pages
        if not text:
            continue

        documents.append(
            {
                "text": text,
                "metadata": {
                    "source": path.name,
                    "page": page_number,
                },
            }
        )

    return documents


# ---------------------------------------------------------
# TEST THE PDF LOADER
# ---------------------------------------------------------

if __name__ == "__main__":

    # IMPORTANT:
    # Replace YOUR_RESUME.pdf with your actual PDF filename.
    pdf_path = r"C:\Users\sudar\OneDrive\Documents\P\ML-Project\data\resume\GRM.pdf"

    try:
        documents = load_pdf(pdf_path)

        print("\n" + "=" * 60)
        print("RAG PDF LOADER TEST")
        print("=" * 60)

        print(f"\nPDF: {Path(pdf_path).name}")
        print(f"Pages extracted: {len(documents)}")

        for document in documents:

            print("\n" + "=" * 60)
            print(f"Page: {document['metadata']['page']}")
            print("=" * 60)

            # Display first 500 characters
            print(document["text"][:500])

        print("\n" + "=" * 60)
        print("PDF LOADING SUCCESSFUL ✅")
        print("=" * 60)

    except Exception as e:

        print("\n❌ ERROR")
        print("-" * 60)
        print(e)