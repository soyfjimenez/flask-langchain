import os
import json

MEMORY_DIR = 'chat_memory'

def get_chat_memory(chat_id):
    file_path = os.path.join(MEMORY_DIR, f'chat_{chat_id}.json')

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f'🚀 No file found at {file_path}')
        return []

    # Try to read the file and print its contents
    try:
        with open(file_path, 'r') as f:
            chat_data = json.load(f)
            return chat_data
    except Exception as e:
        print(f'🚀 Error reading chat memory file: {e}')
        return []

def save_chat_memory(chat_id, chat_history):
    file_path = os.path.join(MEMORY_DIR, f'chat_{chat_id}.json')
    with open(file_path, 'w') as f:
        json.dump(chat_history, f)
