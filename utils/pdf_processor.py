import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Directory to store the FAISS vector store and documents
DB_FAISS_PATH = 'embeddings'
DOCUMENTS_DIR = 'documents'


# Verificar si el directorio donde se almacenará el FAISS index existe
# if not os.path.exists(DB_FAISS_PATH):
#     os.makedirs(DB_FAISS_PATH)


# Ensure the necessary directories exist
if not os.path.exists(DOCUMENTS_DIR):
    os.makedirs(DOCUMENTS_DIR)

# Step 1: Load PDF files and extract text
def load_pdfs():
    texts = []
    for filename in os.listdir(DOCUMENTS_DIR):
        if filename.endswith('.pdf'):
            file_path = os.path.join(DOCUMENTS_DIR, filename)
            # Load PDF using PyPDFLoader
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            texts.extend(documents)
    return texts

# Step 2: Chunk the PDF text into smaller segments
def chunk_text(texts):
    # Use RecursiveCharacterTextSplitter to split the PDF content into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    chunks = text_splitter.split_documents(texts)
    return chunks

# Step 3: Create and save FAISS index with embeddings
def create_faiss_index():
    if os.path.exists(DB_FAISS_PATH):  # Check if the FAISS index already exists
        print("FAISS index already exists. Skipping embedding process.")
        return

    print("FAISS index not found. Processing documents and creating FAISS index...")

    # Load the PDFs
    documents = load_pdfs()

    if not documents:
        print("No PDFs found in the documents directory.")
        return

    # Split the text into smaller chunks
    chunks = chunk_text(documents)

    # Create OpenAI embeddings
    embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small')

    # Create a FAISS vector store and embed the chunks
    faiss_index = FAISS.from_documents(chunks, embeddings)

    # Save the FAISS vector store locally
    faiss_index.save_local(DB_FAISS_PATH)
    print("FAISS index created and saved to:", DB_FAISS_PATH)

# Step 4: Load FAISS index for retrieval
def load_knowledge_base():
    # Ensure FAISS index exists before loading
    if not os.path.exists(DB_FAISS_PATH):
        raise FileNotFoundError(f"No FAISS index found at {DB_FAISS_PATH}. Run the FAISS index creation process first.")

    print("Loading OpenAI Embeddings...")
    embeddings = OpenAIEmbeddings()

    try:
        print("Loading FAISS index...")
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

        return db
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        raise e

# Step 5: Retrieve similar documents based on user query
def retrieve_documents(query, db, top_k=5):
    # Perform similarity search using the FAISS vector store
    similar_documents = db.similarity_search(query, top_k=top_k)
    return similar_documents

# Helper function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == '__main__':
    # Replace with your OpenAI API key

    # Step to create and save FAISS index (run only once)
    create_faiss_index()

    # Once the FAISS index is created, you can load it and retrieve documents
    db = load_knowledge_base()

    # Example query
    query = "What is the main topic of the document?"
    retrieved_docs = retrieve_documents(query, db)

    # Format and print retrieved document chunks
    print("Retrieved Documents:")
    print(format_docs(retrieved_docs))