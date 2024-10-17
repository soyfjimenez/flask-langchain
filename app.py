from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from utils.memory_manager import get_chat_memory, save_chat_memory
from utils.pdf_processor import create_faiss_index, load_knowledge_base, retrieve_documents, format_docs
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from utils.prompt_maker import STANDALONE_QUESTION_PROMPT

# Load environment variables (API keys, etc.)
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Replace with your actual OpenAI API key
api_key = os.getenv('OPENAI_API_KEY')
# os.getenv('LANGSMITH_PROJECT')
# os.getenv('LANGSMITH_TRACING_V2')

# Create FAISS index if it doesn't exist (called when the app starts)
create_faiss_index(api_key)

# Load the FAISS index (knowledge base) for document retrieval
faiss_index = load_knowledge_base(api_key)

# Initialize OpenAI LLM (GPT-3.5 Turbo)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=api_key)

# Define the prompt template
def load_prompt():
    prompt = """
        You need to answer the question based only on the content from the PDF provided.
        Given below is the context and question of the user:

        Context: {context}
        Question: {question}

        If the answer is not in the PDF, respond with "I do not know what you're asking about."
    """
    return ChatPromptTemplate.from_template(prompt)

# Initialize the prompt template
prompt_template = load_prompt()

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('user_message')
        chat_id = data.get('chat_id')

        # Check if user_message and chat_id are provided
        if not user_message or not chat_id:
            return jsonify({'error': 'user_message and chat_id are required'}), 400

        # Retrieve conversation history from JSON memory
        chat_history = get_chat_memory(chat_id)


        if len(chat_history) == 0:
            docs = retrieve_documents(user_message, faiss_index)

            # Format the retrieved documents as context
            context = format_docs(docs)

            # If no relevant documents are found, return a "no result" message
            if not context:
                response = "No relevant information found in the PDFs."
            else:
                # Prepare the prompt for the LLM (OpenAI GPT-3.5)
                formatted_prompt = prompt_template.format(context=context, question=user_message)

                # Process the user message with LangChain LLM (OpenAI GPT-3.5)
                response = llm.invoke(formatted_prompt)

            # Extract the content from AIMessage object
            response_text = response.content

            # Extract the content from AIMessage object
            chat_history.append({'user': user_message, 'assistant': response_text})
            save_chat_memory(chat_id, chat_history)
            return jsonify({
                'response': response_text
            }), 200

        # Format the chat history for the prompt
        formatted_chat_history = "\n".join([f"Human: {msg['user']}\nAI: {msg['assistant']}" for msg in chat_history])

        # Generate a standalone question using the history and user input
        standalone_inputs = {
            "chat_history": formatted_chat_history,
            "question": user_message
        }
        standalone_question_result = STANDALONE_QUESTION_PROMPT.invoke(standalone_inputs).to_string()

        # Extract the content from AIMessage object
        standalone_question_content = llm.invoke(standalone_question_result).content

        # Retrieve relevant document chunks from FAISS based on the user query
        docs = retrieve_documents(standalone_question_content, faiss_index)

        # Format the retrieved documents as context
        context = format_docs(docs)

        # If no relevant documents are found, return a "no result" message
        if not context:
            response_text = "No relevant information found in the PDFs."
        else:
            # Prepare the prompt for the LLM (OpenAI GPT-3.5)
            final_prompt = f"Answer the question based only on the following context:\n\n{context}\n\nQuestion: {standalone_question_content}"

            # Process the user message with LangChain LLM (OpenAI GPT-3.5)
            ai_response = llm.invoke(final_prompt)  # Use invoke method to avoid deprecation
            response_text = ai_response.content  # Extract the text content from AIMessage

        # serialized_docs = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]

        # Update chat memory with the user's message and LLM's response
        chat_history.append({'user': user_message, 'assistant': response_text})
        save_chat_memory(chat_id, chat_history)

        return jsonify({
            'response': response_text
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
