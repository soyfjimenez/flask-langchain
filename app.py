from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from utils.memory_manager import get_chat_memory, save_chat_memory
from utils.pdf_processor_qdrant import create_qdrant_index
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from agents.reAct_agent import process_user_input

# Load environment variables (API keys, etc.)
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Replace with your actual OpenAI API key
api_key = os.getenv('OPENAI_API_KEY')
# os.getenv('LANGSMITH_PROJECT')
# os.getenv('LANGSMITH_TRACING_V2')

# Ensure the Qdrant index is created when the app starts
create_qdrant_index()


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

# # Initialize the prompt template
# prompt_template = load_prompt()

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

        # Process the user message using the agent executor
        response_text = process_user_input(user_message)

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
