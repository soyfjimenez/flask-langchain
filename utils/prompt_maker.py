from langchain.prompts import ChatPromptTemplate

# Define a prompt template for generating standalone questions
standalone_question_template = """
[INST]
Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.
If 'Follow Up Input' is not relevant to chat_history, then fill 'No Context' in Standalone question and leave empty

Example:
Chat History:
Human: How is Mahomes doing?
AI: Mahomes is not looking great, bench him.
Follow Up Input: Who should I replace him with?
Standalone Question: Who should I replace Mahomes with?

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:
[/INST]
"""

STANDALONE_QUESTION_PROMPT = ChatPromptTemplate.from_template(standalone_question_template)