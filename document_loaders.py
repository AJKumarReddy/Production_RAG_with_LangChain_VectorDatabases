from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()


def pdf_loader(file_path: str):
    """Load a PDF file."""

    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        print(f"Loaded {len(documents)} page(s)\n")

        for doc in documents:
            print(doc.metadata)
            # print(doc.page_content)

    except Exception as e:
        print(f"Error loading PDF file: {e}")
        return None


if __name__ == "__main__":
    print("Loading PDF file...")
    pdf_loader("./docs/langchain_demo.pdf")