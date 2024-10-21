# tools.py

import json
from langchain_core.tools import Tool
from utils.pdf_processor_qdrant import retrieve_documents, format_docs, load_knowledge_base
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# OpenAI API key
api_key = os.getenv('OPENAI_API_KEY')

# Load the knowledge base (Qdrant vector store)
db = load_knowledge_base(api_key)

# Update the file path to access the socks.json file inside the documents folder
file_path = os.path.join('documents', 'socks.json')

# Load the JSON data once when the module is imported
with open(file_path, 'r') as f:
    json_data = json.load(f)

def search_json(query):
    """Search the JSON data based on the user's query and return relevant information."""
    # Convert the query to lowercase for case-insensitive matching
    query_lower = query.lower()

    # Initialize an empty list to collect relevant products
    relevant_items = []

    # Iterate over each product in the JSON data
    for item in json_data:
        if isinstance(item, dict):
            # Check if the query mentions the product's ref code
            if 'ref' in item and item['ref'].lower() in query_lower:
                relevant_items.append(item)
                continue  # Proceed to the next item

            # Check if the query mentions the category
            if 'cat' in item and item['cat'].lower() in query_lower:
                relevant_items.append(item)
                continue

            # Check if the query matches any titles or descriptions in translations
            if 'translations' in item:
                for translation in item['translations']:
                    if 'title' in translation and translation['title'].lower() in query_lower:
                        relevant_items.append(item)
                        break
                    if 'description' in translation and translation['description'].lower() in query_lower:
                        relevant_items.append(item)
                        break

    if not relevant_items:
        return "No matching products found in the JSON data."

    # Prepare a response by extracting relevant details from the matched products
    response = ""
    for product in relevant_items:
        product_ref = product.get('ref', 'N/A')
        product_cat = product.get('cat', 'N/A')

        # Extract translations in English
        english_translation = next(
            (t for t in product.get('translations', []) if t.get('language') == 'en'), {}
        )
        product_title = english_translation.get('title', 'N/A')
        product_description = english_translation.get('description', 'N/A')
        product_composition = english_translation.get('composition', 'N/A')

        # Extract pricing information
        pricing_info = ""
        for price in product.get('prices', []):
            qty = price.get('qty', 'N/A')
            price_per_unit = price.get('price', 'N/A')
            pricing_info += f"Quantity: {qty}, Price per unit: {price_per_unit}\n"

        response += (
            f"Product Ref: {product_ref}\n"
            f"Category: {product_cat}\n"
            f"Title: {product_title}\n"
            f"Description: {product_description}\n"
            f"Composition: {product_composition}\n"
            f"Pricing:\n{pricing_info}\n"
            "---------------------\n"
        )

    return response

# Define the JSON search tool
search_json_tool = Tool(
    name="SearchJSON",
    func=search_json,
    description=(
        "Use this tool to search for specific references or IDs in the attached JSON data. "
        "Ideal when the user asks for details based on a reference number or ID."
    )
)

def search_vector_database(query):
    """Search the Qdrant vector database and return relevant documents."""
    retrieved_docs = retrieve_documents(query, db)
    if not retrieved_docs:
        return "No relevant information found in the PDFs."
    formatted_docs = format_docs(retrieved_docs)
    return formatted_docs

# Define the Qdrant search tool
qdrant_search_tool = Tool(
    name="VectorDatabaseSearch",
    func=search_vector_database,
    description=(
        "Use this tool to search the content of the PDFs stored in the vector database. "
        "Ideal for answering questions about the content of the documents."
    )
)

# List of tools
tools = [search_json_tool, qdrant_search_tool]
