# pdf_processor.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Directory to store the documents
DOCUMENTS_DIR = 'documents'

# Ensure the necessary directories exist
if not os.path.exists(DOCUMENTS_DIR):
    os.makedirs(DOCUMENTS_DIR)

# Qdrant credentials
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
QDRANT_URL = os.getenv('QDRANT_URL')

# Collection name in Qdrant
COLLECTION_NAME = os.getenv('COLLECTION_NAME')  # Replace with your desired collection name

API_KEY=os.getenv('OPENAI_API_KEY')

# Step 1: Load PDF files and extract text
def load_pdfs():
    texts = []
    for filename in os.listdir(DOCUMENTS_DIR):
        if filename.endswith('.pdf'):
            file_path = os.path.join(DOCUMENTS_DIR, filename)
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            texts.extend(documents)
    return texts

# Step 2: Chunk the PDF text into smaller segments
def chunk_text(texts):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    chunks = text_splitter.split_documents(texts)
    return chunks

# Step 3: Create and save Qdrant index with embeddings
def create_qdrant_index():
    # OpenAI Embeddings
    embeddings = OpenAIEmbeddings(api_key=API_KEY)

    # Qdrant Client
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

    # Check if the collection already exists
    existing_collections = qdrant_client.get_collections().collections
    if any(collection.name == COLLECTION_NAME for collection in existing_collections):
        print("Qdrant collection already exists. Skipping embedding process.")
        return

    print("Qdrant collection not found. Processing documents and creating Qdrant collection...")

    # Load the PDFs
    documents = load_pdfs()

    if not documents:
        print("No PDFs found in the documents directory.")
        return

    # Split the text into smaller chunks
    chunks = chunk_text(documents)

    # Create a Qdrant vector store and embed the chunks
    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME
    )

    print("Qdrant collection created and documents embedded.")

# Step 4: Load Qdrant index for retrieval
def load_knowledge_base(api_key):
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

    # Check if the collection exists
    existing_collections = qdrant_client.get_collections().collections
    if not any(collection.name == COLLECTION_NAME for collection in existing_collections):
        raise ValueError(f"No Qdrant collection named '{COLLECTION_NAME}' found. Run the index creation process first.")

    embeddings = OpenAIEmbeddings(api_key=api_key)

    # Load the Qdrant vector store
    db = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )

    return db

# Step 5: Retrieve similar documents based on user query
def retrieve_documents(query, db, top_k=5):
    similar_documents = db.similarity_search(query, k=top_k)
    return similar_documents

# Helper function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == '__main__':
    # Ensure the Qdrant index is created
    create_qdrant_index()
